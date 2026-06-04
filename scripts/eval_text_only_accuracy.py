#!/usr/bin/env python3
"""Batch evaluation for text-only FrozenLake planning."""

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
import sys
sys.path.insert(0, str(SRC_DIR))

from model import GridCoTForConditionalGeneration
from constants import GRID_SYSTEM_MESSAGE_TEXT_ONLY, build_grid_user_prompt, infer_grid_size_from_text
from inference import clean_output, extract_answer, build_suppress_token_ids, SuppressPadTokens


MOVE_PATTERN = re.compile(r"go\s+(left|right|up|down)", re.IGNORECASE)
ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)
GOAL_PATTERN = re.compile(
    r"(?:goal(?: is| =)?(?: at)?|trying to reach|reach)\s*\((\d+),\s*(\d+)\)",
    re.IGNORECASE,
)
FROM_TO_PATTERN = re.compile(
    r"from\s*\((\d+),\s*(\d+)\)\s*to\s*\((\d+),\s*(\d+)\)",
    re.IGNORECASE,
)
START_PATTERN = re.compile(
    r"(?:start(?:ing position)?(?: is| =)?|I start at|starting at)\s*\((\d+),\s*(\d+)\)",
    re.IGNORECASE,
)
HOLES_LINE_PATTERN = re.compile(
    r"holes?(?: I must avoid)?\s*:\s*((?:\(\d+,\s*\d+\)\s*,?\s*)+)",
    re.IGNORECASE,
)
COORD_PATTERN = re.compile(r"\((\d+),\s*(\d+)\)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means full dataset")
    parser.add_argument("--use-textual-state-desc", action="store_true")
    return parser.parse_args()


def parse_start(text):
    match = START_PATTERN.search(text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return (0, 0)


def parse_goal(text):
    match = GOAL_PATTERN.search(text)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = FROM_TO_PATTERN.search(text)
    if match:
        return int(match.group(3)), int(match.group(4))
    return None


def parse_holes(text):
    match = HOLES_LINE_PATTERN.search(text)
    if not match:
        return set()
    return {tuple(map(int, xy)) for xy in COORD_PATTERN.findall(match.group(1))}


def infer_bounds(start, goal, holes):
    coords = [start, goal, *holes]
    max_row = max(r for r, _ in coords)
    max_col = max(c for _, c in coords)
    return max_row + 1, max_col + 1


def apply_move(position, move):
    row, col = position
    if move == "up":
        return row - 1, col
    if move == "down":
        return row + 1, col
    if move == "left":
        return row, col - 1
    if move == "right":
        return row, col + 1
    raise ValueError(f"Unsupported move: {move}")


def normalize_reference_answer(text):
    match = ANSWER_PATTERN.search(text)
    if not match:
        return None
    answer = match.group(1).strip()
    lower = answer.lower()
    if "fail to find a valid path" in lower:
        return "fail to find a valid path"
    moves = re.findall(r"go\s+(?:left|right|up|down)", lower)
    if not moves:
        return answer
    return " -> ".join(move.strip() for move in moves)


def simulate_answer(reference_text, predicted_answer):
    start = parse_start(reference_text)
    goal = parse_goal(reference_text)
    if goal is None:
        raise ValueError("Could not parse goal coordinates from reference text")
    holes = parse_holes(reference_text)
    rows, cols = infer_bounds(start, goal, holes)

    if predicted_answer is None:
        return {
            "status": "no_answer",
            "reason": "answer extraction failed",
            "trajectory": [start],
        }

    if predicted_answer.strip().lower() == "fail to find a valid path":
        return {
            "status": "declared_no_path",
            "reason": "model declared no valid path",
            "trajectory": [start],
        }

    moves = [m.lower() for m in MOVE_PATTERN.findall(predicted_answer)]
    position = start
    trajectory = [position]

    for step_id, move in enumerate(moves, start=1):
        next_pos = apply_move(position, move)

        # FrozenLake semantics: out-of-bounds action keeps the agent in place.
        if not (0 <= next_pos[0] < rows and 0 <= next_pos[1] < cols):
            next_pos = position
        trajectory.append(next_pos)
        if next_pos in holes:
            return {
                "status": "fell_into_hole",
                "reason": f"step {step_id}: {move} -> {next_pos}",
                "trajectory": trajectory,
            }

        position = next_pos
        if position == goal:
            return {
                "status": "reached_goal",
                "reason": f"step {step_id}: {move} -> {position}",
                "trajectory": trajectory,
            }

    return {
        "status": "stopped_before_goal",
        "reason": f"finished at {position}, goal is {goal}",
        "trajectory": trajectory,
    }


def build_messages(sample, reference, use_textual_state_desc=False):
    rows, cols = infer_grid_size_from_text(reference)
    user_text = build_grid_user_prompt(
        rows=rows,
        cols=cols,
        include_grid_token=False,
        use_textual_state_desc=False,
    )
    image_path = sample["images"][0]
    return [
        {"role": "system", "content": [{"type": "text", "text": GRID_SYSTEM_MESSAGE_TEXT_ONLY}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset_path) as f:
        dataset = json.load(f)
    if args.max_samples and args.max_samples > 0:
        dataset = dataset[: args.max_samples]

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = GridCoTForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    suppress_ids = build_suppress_token_ids(processor.tokenizer)

    records = []
    totals = {
        "num_samples": len(dataset),
        "num_reached_goal": 0,
        "num_exact_match": 0,
        "num_declared_no_path": 0,
        "num_no_answer": 0,
    }
    status_counts = {}

    started = time.time()

    for batch_start in range(0, len(dataset), args.batch_size):
        batch = dataset[batch_start : batch_start + args.batch_size]
        messages_batch = []
        prompts = []
        references = []

        for sample in batch:
            reference = sample["conversations"][-1]["value"]
            messages = build_messages(sample, reference, use_textual_state_desc=args.use_textual_state_desc)
            messages_batch.append(messages)
            references.append(reference)
            prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        image_inputs = []
        for messages in messages_batch:
            batch_images, _ = process_vision_info(messages)
            image_inputs.append(batch_images[0])

        inputs = processor(
            text=prompts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                logits_processor=[SuppressPadTokens(suppress_ids)] if suppress_ids else None,
            )

        for local_idx, sample in enumerate(batch):
            sample_index = batch_start + local_idx
            prompt_len = int(inputs.input_ids[local_idx].ne(processor.tokenizer.pad_token_id).sum().item())
            trimmed = generated[local_idx][prompt_len:]
            raw_output = processor.decode(trimmed, skip_special_tokens=False)
            predicted_answer = extract_answer(raw_output)
            reference = references[local_idx]
            reference_answer = normalize_reference_answer(reference)
            sim = simulate_answer(reference, predicted_answer)

            exact_match = predicted_answer == reference_answer
            reached_goal = sim["status"] == "reached_goal"

            if reached_goal:
                totals["num_reached_goal"] += 1
            if exact_match:
                totals["num_exact_match"] += 1
            if sim["status"] == "declared_no_path":
                totals["num_declared_no_path"] += 1
            if sim["status"] == "no_answer":
                totals["num_no_answer"] += 1
            status_counts[sim["status"]] = status_counts.get(sim["status"], 0) + 1

            records.append(
                {
                    "sample_index": sample_index,
                    "raw_output": raw_output,
                    "clean_output": clean_output(raw_output),
                    "predicted_answer": predicted_answer,
                    "reference_answer": reference_answer,
                    "simulation": sim,
                    "exact_match": exact_match,
                    "reached_goal": reached_goal,
                }
            )

        print(f"processed {min(batch_start + len(batch), len(dataset))}/{len(dataset)}")

    elapsed = time.time() - started
    totals["reached_goal_accuracy"] = totals["num_reached_goal"] / max(totals["num_samples"], 1)
    totals["exact_match_accuracy"] = totals["num_exact_match"] / max(totals["num_samples"], 1)
    totals["status_counts"] = status_counts
    totals["elapsed_seconds"] = round(elapsed, 2)

    timestamp = datetime.now(timezone.utc).astimezone().isoformat()
    metadata = {
        "timestamp": timestamp,
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "note": "No dedicated test split was found; this run uses the provided dataset as the evaluation set.",
    }

    with (output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with (output_dir / "summary.json").open("w") as f:
        json.dump(totals, f, ensure_ascii=False, indent=2)
    with (output_dir / "results.json").open("w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    summary_lines = [
        "Text-Only Eval Summary",
        f"DateTime: {timestamp}",
        f"Model: {args.model_path}",
        f"Dataset: {args.dataset_path}",
        f"Samples: {totals['num_samples']}",
        f"ReachedGoalAccuracy: {totals['reached_goal_accuracy']:.4f}",
        f"ExactMatchAccuracy: {totals['exact_match_accuracy']:.4f}",
        f"ReachedGoalCount: {totals['num_reached_goal']}",
        f"ExactMatchCount: {totals['num_exact_match']}",
        f"NoAnswerCount: {totals['num_no_answer']}",
        f"DeclaredNoPathCount: {totals['num_declared_no_path']}",
        f"ElapsedSeconds: {totals['elapsed_seconds']}",
        f"StatusCounts: {json.dumps(status_counts, ensure_ascii=False)}",
        "",
        metadata["note"],
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")

    print(f"saved to {output_dir}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
