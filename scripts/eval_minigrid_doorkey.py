#!/usr/bin/env python3
"""Evaluate MiniGrid DoorKey planning accuracy."""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from constants import (
    ANSWER_END,
    ANSWER_START,
    MINIGRID_SYSTEM_MESSAGE,
    MINIGRID_SYSTEM_MESSAGE_TEXT_ONLY,
    MINIGRID_NAV_SYSTEM_MESSAGE,
    MINIGRID_NAV_SYSTEM_MESSAGE_TEXT_ONLY,
    MINIGRID_NAV_SYSTEM_MESSAGE_STATE_DESC,
    MINIGRID_SYSTEM_MESSAGE_STATE_DESC,
    THINK_END,
    THINK_START,
    build_grid_user_prompt,
)
from inference import (
    GridFormatEnforcer,
    SuppressPadTokens,
    build_suppress_token_ids,
    clean_output,
)
from model import GridCoTForConditionalGeneration


DOORKEY_ACTION_PATTERN = re.compile(
    r"(turn\s+left|turn\s+right|go\s+forward|pickup\s+key|toggle\s+door)",
    re.IGNORECASE,
)
NAV_ACTION_PATTERN = re.compile(
    r"(turn\s+left|turn\s+right|go\s+forward)",
    re.IGNORECASE,
)
ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)
START_PATTERN = re.compile(
    r"(?:starting position is|I start at|starting at)\s*\((\d+),\s*(\d+)\)",
    re.IGNORECASE,
)
START_DIR_PATTERN = re.compile(
    r"starting position is\s*\(\d+,\s*\d+\),\s*facing\s*(north|east|south|west)",
    re.IGNORECASE,
)
GOAL_PATTERN = re.compile(r"goal(?: is| =)?(?: at)?\s*\((\d+),\s*(\d+)\)", re.IGNORECASE)
KEY_PATTERN = re.compile(r"key(?: is| =)?(?: at)?\s*\((\d+),\s*(\d+)\)", re.IGNORECASE)
DOOR_PATTERN = re.compile(r"door(?: is| =)?(?: at)?\s*\((\d+),\s*(\d+)\)", re.IGNORECASE)
WALLS_LINE_PATTERN = re.compile(
    r"wall tiles?(?: I must avoid)?\s*:\s*((?:\(\d+,\s*\d+\)\s*,?\s*)+)",
    re.IGNORECASE,
)
GRID_SIZE_PATTERN = re.compile(r"(\d+)x(\d+)\s+grid", re.IGNORECASE)
COORD_PATTERN = re.compile(r"\((\d+),\s*(\d+)\)")
DIR_TO_ID = {"north": 0, "east": 1, "south": 2, "west": 3}
DIR_TO_TEXT = {v: k for k, v in DIR_TO_ID.items()}
FORWARD_DELTAS = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}


def infer_env_type(sample, reference_text):
    meta = sample.get("meta", {})
    env_type = meta.get("env_type")
    if env_type:
        return env_type
    lowered = (reference_text or "").lower()
    if "minigrid-style navigation" in lowered:
        return "minigrid_nav"
    return "minigrid"


def get_action_pattern(env_type):
    return NAV_ACTION_PATTERN if env_type == "minigrid_nav" else DOORKEY_ACTION_PATTERN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--enforce-format", type=str, default="False")
    parser.add_argument("--use-textual-state-desc", type=str, default="False")
    parser.add_argument("--use-grid-tokens", type=str, default="True",
                        help="Set False for text-only models (no <grid_token> in prompt/system message)")
    parser.add_argument("--ignore-mismatched-sizes", type=str, default="False")
    parser.add_argument("--model-base", type=str, default=None)
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


def parse_env(sample, reference_text):
    meta = sample.get("meta", {})
    env_type = infer_env_type(sample, reference_text)
    rows = meta.get("rows")
    cols = meta.get("cols")
    if rows is None or cols is None:
        size_match = GRID_SIZE_PATTERN.search(reference_text)
        if not size_match:
            raise ValueError("Could not parse grid size")
        rows, cols = int(size_match.group(1)), int(size_match.group(2))

    if "start" in meta:
        start = tuple(meta["start"])
    else:
        match = START_PATTERN.search(reference_text)
        start = (int(match.group(1)), int(match.group(2))) if match else (0, 0)

    if "start_dir" in meta:
        start_dir = DIR_TO_ID[str(meta["start_dir"]).lower()]
    else:
        match = START_DIR_PATTERN.search(reference_text)
        start_dir = DIR_TO_ID[match.group(1).lower()] if match else 0

    if "goal" in meta:
        goal = tuple(meta["goal"])
    else:
        match = GOAL_PATTERN.search(reference_text)
        if match is None:
            raise ValueError("Could not parse goal")
        goal = (int(match.group(1)), int(match.group(2)))

    key_pos = None
    door_pos = None
    if env_type != "minigrid_nav":
        if "key" in meta:
            key_pos = tuple(meta["key"])
        else:
            match = KEY_PATTERN.search(reference_text)
            key_pos = (int(match.group(1)), int(match.group(2))) if match else None

        if "door" in meta:
            door_pos = tuple(meta["door"])
        else:
            match = DOOR_PATTERN.search(reference_text)
            door_pos = (int(match.group(1)), int(match.group(2))) if match else None

    if "walls" in meta:
        walls = {tuple(x) for x in meta["walls"]}
    else:
        match = WALLS_LINE_PATTERN.search(reference_text)
        walls = {tuple(map(int, xy)) for xy in COORD_PATTERN.findall(match.group(1))} if match else set()

    return {
        "env_type": env_type,
        "rows": rows,
        "cols": cols,
        "start": start,
        "start_dir": start_dir,
        "goal": goal,
        "key_pos": key_pos,
        "door_pos": door_pos,
        "walls": walls,
    }


def normalize_action_answer(text, env_type):
    if text is None:
        return None
    actions = [a.lower().strip() for a in get_action_pattern(env_type).findall(text)]
    if not actions:
        return None
    return " -> ".join(actions)


def extract_answer(text, env_type):
    text = clean_output(text)
    # Find the LAST <answer> and the LAST </answer>
    # Model sometimes emits spurious <answer> tags inside CoT
    last_open = text.rfind("<answer>")
    last_close = text.rfind("</answer>")
    if last_open >= 0 and last_close > last_open:
        content = text[last_open + len("<answer>"):last_close].strip()
        return normalize_action_answer(content, env_type)
    # Fallback: standard regex
    match = ANSWER_PATTERN.search(text)
    if not match:
        return None
    return normalize_action_answer(match.group(1), env_type)


def normalize_reference_answer(text, env_type):
    match = ANSWER_PATTERN.search(text)
    if not match:
        return None
    return normalize_action_answer(match.group(1), env_type)


def build_messages(sample, use_textual_state_desc=False, use_grid_tokens=True):
    reference = sample["conversations"][-1]["value"]
    meta = sample.get("meta", {})
    rows = meta.get("rows")
    cols = meta.get("cols")
    env_type = infer_env_type(sample, reference)
    include_grid_token = use_grid_tokens and not use_textual_state_desc
    user_text = build_grid_user_prompt(
        rows=rows,
        cols=cols,
        include_grid_token=include_grid_token,
        use_textual_state_desc=use_textual_state_desc,
        env_type=env_type,
    )
    if env_type == "minigrid_nav":
        if use_textual_state_desc:
            system_text = MINIGRID_NAV_SYSTEM_MESSAGE_STATE_DESC
        elif use_grid_tokens:
            system_text = MINIGRID_NAV_SYSTEM_MESSAGE
        else:
            system_text = MINIGRID_NAV_SYSTEM_MESSAGE_TEXT_ONLY
    else:
        if use_textual_state_desc:
            system_text = MINIGRID_SYSTEM_MESSAGE_STATE_DESC
        elif use_grid_tokens:
            system_text = MINIGRID_SYSTEM_MESSAGE
        else:
            system_text = MINIGRID_SYSTEM_MESSAGE_TEXT_ONLY
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


def apply_action(state, action, env):
    pos = state["pos"]
    direction = state["dir"]
    carrying_key = state["carrying_key"]
    key_present = state["key_present"]
    door_open = state["door_open"]
    rows = env["rows"]
    cols = env["cols"]
    walls = env["walls"]
    key_pos = env["key_pos"]
    door_pos = env["door_pos"]

    invalid = False

    if action == "turn left":
        direction = (direction - 1) % 4
    elif action == "turn right":
        direction = (direction + 1) % 4
    elif action == "go forward":
        dr, dc = FORWARD_DELTAS[direction]
        nr, nc = pos[0] + dr, pos[1] + dc
        if not (0 <= nr < rows and 0 <= nc < cols):
            invalid = True
        elif (nr, nc) in walls:
            invalid = True
        elif door_pos is not None and (nr, nc) == door_pos and not door_open:
            invalid = True
        else:
            pos = (nr, nc)
    elif action == "pickup key":
        dr, dc = FORWARD_DELTAS[direction]
        front = (pos[0] + dr, pos[1] + dc)
        if key_pos is None or not key_present or carrying_key or front != key_pos:
            invalid = True
        else:
            carrying_key = True
            key_present = False
    elif action == "toggle door":
        dr, dc = FORWARD_DELTAS[direction]
        front = (pos[0] + dr, pos[1] + dc)
        if door_pos is None or front != door_pos or door_open or not carrying_key:
            invalid = True
        else:
            door_open = True
    else:
        raise ValueError(f"Unsupported action: {action}")

    return {
        "pos": pos,
        "dir": direction,
        "carrying_key": carrying_key,
        "key_present": key_present,
        "door_open": door_open,
    }, invalid


def simulate_answer(sample, predicted_answer):
    reference = sample["conversations"][-1]["value"]
    env = parse_env(sample, reference)
    state = {
        "pos": env["start"],
        "dir": env["start_dir"],
        "carrying_key": False,
        "key_present": env["key_pos"] is not None,
        "door_open": False,
    }
    trajectory = [
        {
            "pos": list(state["pos"]),
            "dir": DIR_TO_TEXT[state["dir"]],
            "carrying_key": state["carrying_key"],
            "key_present": state["key_present"],
            "door_open": state["door_open"],
        }
    ]

    if predicted_answer is None:
        return {
            "status": "no_answer",
            "reason": "answer extraction failed",
            "trajectory": trajectory,
            "invalid_steps": [],
        }

    actions = [a.lower().strip() for a in get_action_pattern(env["env_type"]).findall(predicted_answer)]
    invalid_steps = []

    for step_id, action in enumerate(actions, start=1):
        state, invalid = apply_action(state, action, env)
        if invalid:
            invalid_steps.append(
                {
                    "step": step_id,
                    "action": action,
                    "pos": list(state["pos"]),
                    "dir": DIR_TO_TEXT[state["dir"]],
                }
            )
        trajectory.append(
            {
                "pos": list(state["pos"]),
                "dir": DIR_TO_TEXT[state["dir"]],
                "carrying_key": state["carrying_key"],
                "key_present": state["key_present"],
                "door_open": state["door_open"],
            }
        )
        if state["pos"] == env["goal"]:
            return {
                "status": "reached_goal",
                "reason": f"step {step_id}: {action} -> {state['pos']}",
                "trajectory": trajectory,
                "invalid_steps": invalid_steps,
            }

    return {
        "status": "stopped_before_goal",
        "reason": f"finished at {state['pos']}, goal is {env['goal']}",
        "trajectory": trajectory,
        "invalid_steps": invalid_steps,
    }


def init_metrics():
    return {
        "num_samples": 0,
        "num_reached_goal": 0,
        "num_no_answer": 0,
        "status_counts": {},
    }


def main():
    args = parse_args()
    enforce_format = _parse_bool(args.enforce_format)
    use_textual_state_desc = _parse_bool(args.use_textual_state_desc)
    use_grid_tokens = _parse_bool(args.use_grid_tokens)
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
    # enforce_format 参数用于每次 generate 前重新构造（状态机不可复用）
    _enforce_format_kwargs = dict(
        think_start_id=_single_token_id(processor.tokenizer, THINK_START),
        think_end_id=_single_token_id(processor.tokenizer, THINK_END),
        answer_start_id=_single_token_id(processor.tokenizer, ANSWER_START),
        answer_end_id=_single_token_id(processor.tokenizer, ANSWER_END),
        eos_id=processor.tokenizer.eos_token_id,
    ) if enforce_format else None

    metrics = init_metrics()
    records = []
    started = time.time()

    batch_starts = range(0, len(dataset), args.batch_size)
    pbar = tqdm(batch_starts, total=len(batch_starts), desc="Evaluating", unit="batch")

    for batch_start in pbar:
        batch = dataset[batch_start : batch_start + args.batch_size]
        messages_batch = []
        prompts = []

        for sample in batch:
            messages = build_messages(sample, use_textual_state_desc=use_textual_state_desc, use_grid_tokens=use_grid_tokens)
            messages_batch.append(messages)
            prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        image_inputs = []
        for messages in messages_batch:
            batch_images, _ = process_vision_info(messages)
            image_inputs.append(batch_images[0])

        inputs = processor(text=prompts, images=image_inputs, padding=True, return_tensors="pt").to("cuda")

        # GridFormatEnforcer 是有状态的状态机，必须每次 generate 前重新实例化
        logits_processors = []
        if suppress_ids:
            logits_processors.append(SuppressPadTokens(suppress_ids))
        if _enforce_format_kwargs is not None:
            logits_processors.append(GridFormatEnforcer(**_enforce_format_kwargs))

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
            reference_text = sample["conversations"][-1]["value"]
            raw_output = processor.decode(trimmed, skip_special_tokens=False)
            env_type = infer_env_type(sample, reference_text)
            predicted_answer = extract_answer(raw_output, env_type)
            reference_answer = normalize_reference_answer(reference_text, env_type)
            sim = simulate_answer(sample, predicted_answer)
            reached_goal = sim["status"] == "reached_goal"

            metrics["num_samples"] += 1
            if reached_goal:
                metrics["num_reached_goal"] += 1
            if sim["status"] == "no_answer":
                metrics["num_no_answer"] += 1
            metrics["status_counts"][sim["status"]] = metrics["status_counts"].get(sim["status"], 0) + 1

            records.append(
                {
                    "sample_index": sample_index,
                    "raw_output": raw_output,
                    "clean_output": clean_output(raw_output),
                    "predicted_answer": predicted_answer,
                    "reference_answer": reference_answer,
                    "simulation": sim,
                    "reached_goal": reached_goal,
                    "image_path": sample["images"][0],
                    "meta": sample.get("meta", {}),
                }
            )

        processed = min(batch_start + len(batch), len(dataset))
        reached_acc = metrics["num_reached_goal"] / max(metrics["num_samples"], 1)
        pbar.set_postfix(processed=f"{processed}/{len(dataset)}", reached_acc=f"{reached_acc:.3f}")

    elapsed = time.time() - started
    metrics["reached_goal_accuracy"] = metrics["num_reached_goal"] / max(metrics["num_samples"], 1)
    metrics["elapsed_seconds"] = round(elapsed, 2)

    timestamp = datetime.now(timezone.utc).astimezone().isoformat()
    env_name = "minigrid_doorkey"
    if dataset:
        env_name = infer_env_type(dataset[0], dataset[0]["conversations"][-1]["value"])
    metadata = {
        "timestamp": timestamp,
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "environment": env_name,
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"overall": metrics}, f, ensure_ascii=False, indent=2)
    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    summary_title = "MiniGrid Nav Eval Summary" if env_name == "minigrid_nav" else "MiniGrid DoorKey Eval Summary"
    summary_lines = [
        summary_title,
        f"DateTime: {timestamp}",
        f"Model: {args.model_path}",
        f"Dataset: {args.dataset_path}",
        f"TotalSamples: {metrics['num_samples']}",
        f"ReachedGoalAccuracy: {metrics['reached_goal_accuracy']:.4f}",
        f"NoAnswer: {metrics['num_no_answer']}",
        f"StatusCounts: {json.dumps(metrics['status_counts'], ensure_ascii=False)}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"saved to {output_dir}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
