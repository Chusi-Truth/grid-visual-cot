#!/usr/bin/env python3
"""Evaluate FrozenLake planning accuracy and report size-bucket metrics."""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from peft import PeftModel
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from constants import (
    GRID_SYSTEM_MESSAGE,
    GRID_SYSTEM_MESSAGE_STATE_DESC,
    THINK_START,
    THINK_END,
    ANSWER_START,
    ANSWER_END,
    build_grid_user_prompt,
    infer_grid_size_from_text,
)
from inference import (
    clean_output,
    extract_answer,
    build_suppress_token_ids,
    SuppressPadTokens,
    GridFormatEnforcer,
)
from model import GridCoTForConditionalGeneration


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
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--threshold-area", type=int, default=20)
    parser.add_argument("--enforce-format", type=str, default="True")
    parser.add_argument("--use-textual-state-desc", type=str, default="False")
    parser.add_argument("--ignore-mismatched-sizes", type=str, default="False")
    parser.add_argument(
        "--model-base",
        type=str,
        default=None,
        help="Optional base model path. If set, --model-path is treated as a LoRA checkpoint dir and merged in-memory.",
    )
    return parser.parse_args()


def _parse_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _single_token_id(tokenizer, token_text):
    ids = tokenizer(token_text, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError(f"Token {token_text} is expected to map to exactly one id, got {ids}")
    return ids[0]


def _build_generation_eos_ids(tokenizer):
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)

    # Stop on common terminators to avoid long trailing garbage.
    for tok in [ANSWER_END, "<|im_end|>", "<|endoftext|>"]:
        ids = tokenizer(tok, add_special_tokens=False).input_ids
        if len(ids) == 1:
            eos_ids.add(ids[0])

    eos_ids = sorted(eos_ids)
    if not eos_ids:
        return tokenizer.eos_token_id
    return eos_ids if len(eos_ids) > 1 else eos_ids[0]


def _has_adapter_weights(model_path):
    return any(
        os.path.exists(os.path.join(model_path, name))
        for name in ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]
    )


def load_processor_and_model(model_path, model_base=None, ignore_mismatched_sizes=False):
    if model_base is None:
        processor = AutoProcessor.from_pretrained(model_path)
        model = GridCoTForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=None,
            low_cpu_mem_usage=False,
            attn_implementation="flash_attention_2",
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )
        model = model.to("cuda")
        model.eval()
        return processor, model

    # Runtime merge mode: load tokenizer/processor from LoRA dir if possible.
    try:
        processor = AutoProcessor.from_pretrained(model_path)
    except Exception:
        processor = AutoProcessor.from_pretrained(model_base)

    model = GridCoTForConditionalGeneration.from_pretrained(
        model_base,
        torch_dtype=torch.float16,
        device_map=None,
        low_cpu_mem_usage=False,
        attn_implementation="flash_attention_2",
        ignore_mismatched_sizes=ignore_mismatched_sizes,
    )

    target_size = len(processor.tokenizer)
    current_size = model.get_input_embeddings().weight.shape[0]
    if current_size < target_size:
        model.resize_token_embeddings(target_size)

    if _has_adapter_weights(model_path):
        model = PeftModel.from_pretrained(model, model_path, device_map="cpu")
        non_lora_path = os.path.join(model_path, "non_lora_state_dict.bin")
        if os.path.exists(non_lora_path):
            non_lora_state = torch.load(non_lora_path, map_location="cpu")
            model.load_state_dict(non_lora_state, strict=False)
        model = model.merge_and_unload()

    model = model.to("cuda")
    model.eval()
    return processor, model


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


def build_messages(sample, reference):
    # Backward-compatible default keeps original behavior.
    return build_messages_with_mode(sample, reference, use_textual_state_desc=False)


def build_messages_with_mode(sample, reference, use_textual_state_desc=False):
    rows, cols = infer_grid_size_from_text(reference)
    user_text = build_grid_user_prompt(
        rows=rows,
        cols=cols,
        include_grid_token=not use_textual_state_desc,
        use_textual_state_desc=use_textual_state_desc,
    )
    if use_textual_state_desc:
        system_text = GRID_SYSTEM_MESSAGE_STATE_DESC
    else:
        system_text = GRID_SYSTEM_MESSAGE
    image_path = sample["images"][0]
    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def bucket_name(rows, cols, threshold_area):
    area = rows * cols
    if area < threshold_area:
        return f"lt_{threshold_area}"
    if area > threshold_area:
        return f"gt_{threshold_area}"
    return f"eq_{threshold_area}"


def init_bucket():
    return {
        "num_samples": 0,
        "num_reached_goal": 0,
        "num_exact_match": 0,
        "num_declared_no_path": 0,
        "num_no_answer": 0,
        "status_counts": {},
    }


def main():
    args = parse_args()
    enforce_format = _parse_bool(args.enforce_format)
    use_textual_state_desc = _parse_bool(args.use_textual_state_desc)
    ignore_mismatched_sizes = _parse_bool(args.ignore_mismatched_sizes)
    if enforce_format and args.batch_size != 1:
        raise ValueError("--enforce-format=True currently requires --batch-size 1.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    if args.max_samples and args.max_samples > 0:
        dataset = dataset[: args.max_samples]

    processor, model = load_processor_and_model(
        args.model_path,
        args.model_base,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
    )
    suppress_ids = build_suppress_token_ids(processor.tokenizer)
    eos_ids_for_generate = _build_generation_eos_ids(processor.tokenizer)
    logits_processors = []
    if suppress_ids:
        logits_processors.append(SuppressPadTokens(suppress_ids))
    if enforce_format:
        logits_processors.append(
            GridFormatEnforcer(
                think_start_id=_single_token_id(processor.tokenizer, THINK_START),
                think_end_id=_single_token_id(processor.tokenizer, THINK_END),
                answer_start_id=_single_token_id(processor.tokenizer, ANSWER_START),
                answer_end_id=_single_token_id(processor.tokenizer, ANSWER_END),
                eos_id=processor.tokenizer.eos_token_id,
            )
        )

    records = []
    totals = init_bucket()
    size_metrics = defaultdict(init_bucket)
    started = time.time()

    batch_starts = range(0, len(dataset), args.batch_size)
    pbar = tqdm(batch_starts, total=len(batch_starts), desc="Evaluating", unit="batch")

    for batch_start in pbar:
        batch = dataset[batch_start : batch_start + args.batch_size]
        messages_batch = []
        prompts = []
        references = []

        for sample in batch:
            reference = sample["conversations"][-1]["value"]
            messages = build_messages_with_mode(
                sample,
                reference,
                use_textual_state_desc=use_textual_state_desc,
            )
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
                logits_processor=logits_processors if logits_processors else None,
                eos_token_id=eos_ids_for_generate,
                pad_token_id=processor.tokenizer.pad_token_id,
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
            rows, cols = infer_grid_size_from_text(reference)
            bucket = bucket_name(rows, cols, args.threshold_area)

            exact_match = predicted_answer == reference_answer
            reached_goal = sim["status"] == "reached_goal"

            for metric in (totals, size_metrics[bucket]):
                metric["num_samples"] += 1
                if reached_goal:
                    metric["num_reached_goal"] += 1
                if exact_match:
                    metric["num_exact_match"] += 1
                if sim["status"] == "declared_no_path":
                    metric["num_declared_no_path"] += 1
                if sim["status"] == "no_answer":
                    metric["num_no_answer"] += 1
                metric["status_counts"][sim["status"]] = metric["status_counts"].get(sim["status"], 0) + 1

            records.append(
                {
                    "sample_index": sample_index,
                    "rows": rows,
                    "cols": cols,
                    "area": rows * cols,
                    "bucket": bucket,
                    "raw_output": raw_output,
                    "clean_output": clean_output(raw_output),
                    "predicted_answer": predicted_answer,
                    "reference_answer": reference_answer,
                    "simulation": sim,
                    "exact_match": exact_match,
                    "reached_goal": reached_goal,
                    "image_path": sample["images"][0],
                }
            )

        processed = min(batch_start + len(batch), len(dataset))
        live_reached_acc = totals["num_reached_goal"] / max(totals["num_samples"], 1)
        live_exact_acc = totals["num_exact_match"] / max(totals["num_samples"], 1)
        pbar.set_postfix(
            processed=f"{processed}/{len(dataset)}",
            reached_acc=f"{live_reached_acc:.3f}",
            exact_acc=f"{live_exact_acc:.3f}",
        )

    elapsed = time.time() - started
    for metric in [totals, *size_metrics.values()]:
        metric["reached_goal_accuracy"] = metric["num_reached_goal"] / max(metric["num_samples"], 1)
        metric["exact_match_accuracy"] = metric["num_exact_match"] / max(metric["num_samples"], 1)

    totals["elapsed_seconds"] = round(elapsed, 2)
    timestamp = datetime.now(timezone.utc).astimezone().isoformat()

    metadata = {
        "timestamp": timestamp,
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "threshold_area": args.threshold_area,
        "bucket_rules": {
            f"lt_{args.threshold_area}": f"rows*cols < {args.threshold_area}",
            f"eq_{args.threshold_area}": f"rows*cols == {args.threshold_area}",
            f"gt_{args.threshold_area}": f"rows*cols > {args.threshold_area}",
        },
    }

    summary = {
        "overall": totals,
        "by_bucket": dict(size_metrics),
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    bucket_lines = []
    for bucket in [f"lt_{args.threshold_area}", f"eq_{args.threshold_area}", f"gt_{args.threshold_area}"]:
        metric = size_metrics.get(bucket)
        if not metric:
            continue
        bucket_lines.extend(
            [
                f"{bucket}.Samples: {metric['num_samples']}",
                f"{bucket}.ReachedGoalAccuracy: {metric['reached_goal_accuracy']:.4f}",
                f"{bucket}.ExactMatchAccuracy: {metric['exact_match_accuracy']:.4f}",
                f"{bucket}.StatusCounts: {json.dumps(metric['status_counts'], ensure_ascii=False)}",
            ]
        )

    summary_lines = [
        "Eval Accuracy By Size Summary",
        f"DateTime: {timestamp}",
        f"Model: {args.model_path}",
        f"Dataset: {args.dataset_path}",
        f"ThresholdArea: {args.threshold_area}",
        f"TotalSamples: {totals['num_samples']}",
        f"TotalReachedGoalAccuracy: {totals['reached_goal_accuracy']:.4f}",
        f"TotalExactMatchAccuracy: {totals['exact_match_accuracy']:.4f}",
        f"TotalStatusCounts: {json.dumps(totals['status_counts'], ensure_ascii=False)}",
        "",
        *bucket_lines,
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"saved to {output_dir}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
