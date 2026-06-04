#!/usr/bin/env python3
"""Evaluate FrozenLake planning with an API-based closed-source model baseline."""

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from constants import (
    GRID_SYSTEM_MESSAGE_TEXT_ONLY,
    build_grid_user_prompt,
    infer_grid_size_from_text,
)


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
PATH_PATTERN = re.compile(
    r"go\s+(?:left|right|up|down)(?:\s*(?:->|=>|,|;|\||/|\\)\s*go\s+(?:left|right|up|down))*",
    flags=re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True, help="API model name, e.g. gpt-4.1")
    parser.add_argument("--api-base-url", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Env var name for API key")
    parser.add_argument("--api-mode", default="chat", choices=["chat", "responses"])
    parser.add_argument("--max-samples", type=int, default=0, help="0 means full dataset")
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    return parser.parse_args()


def clean_output(text: str) -> str:
    """Remove special tokens and normalize whitespace."""
    text = text.replace("<|grid_pad|>", " ")
    text = re.sub(r"<\|[^>]+\|>", " ", text)
    text = text.replace("\ufffd", " ")
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_path_text(text: str) -> Optional[str]:
    """
    Normalize heterogeneous action formats into:
      go right -> go down -> ...
    Supported forms include:
      - go right / right / up / down
      - R L U D
      - 中文方向: 向右/向左/向上/向下/右/左/上/下
    """
    src = text.strip().lower()
    # normalize common separators/symbols
    for sep in ["->", "=>", "→", "|", ",", ";", "/", "\\", "\n", "\t"]:
        src = src.replace(sep, " ")
    src = re.sub(r"\s+", " ", src)

    # ordered pattern (specific first)
    token_pat = re.compile(
        r"go\s+(left|right|up|down)"
        r"|\b(left|right|up|down)\b"
        r"|\b([lrud])\b"
        r"|(向右|向左|向上|向下|右|左|上|下)",
        flags=re.IGNORECASE,
    )

    mapped = []
    for m in token_pat.finditer(src):
        g1, g2, g3, g4 = m.groups()
        if g1:
            mapped.append(g1.lower())
            continue
        if g2:
            mapped.append(g2.lower())
            continue
        if g3:
            mapped.append({"l": "left", "r": "right", "u": "up", "d": "down"}[g3.lower()])
            continue
        if g4:
            mapped.append(
                {
                    "向右": "right",
                    "右": "right",
                    "向左": "left",
                    "左": "left",
                    "向上": "up",
                    "上": "up",
                    "向下": "down",
                    "下": "down",
                }[g4]
            )

    if not mapped:
        return None
    return " -> ".join(f"go {x}" for x in mapped)


def extract_answer(text: str) -> Optional[str]:
    """Extract the final action path from raw model output."""
    text = clean_output(text)

    match = ANSWER_PATTERN.search(text)
    if match:
        answer = normalize_path_text(match.group(1))
        if answer:
            return answer

    candidates = PATH_PATTERN.findall(text)
    if candidates:
        candidates.sort(key=len, reverse=True)
        answer = normalize_path_text(candidates[0])
        if answer:
            return answer
    return None


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


def image_path_to_data_url(image_path: str) -> str:
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    mime, _ = mimetypes.guess_type(str(image_file))
    if not mime:
        mime = "image/jpeg"
    b64 = base64.b64encode(image_file.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_messages(sample, reference):
    rows, cols = infer_grid_size_from_text(reference)
    user_text = build_grid_user_prompt(
        rows=rows,
        cols=cols,
        include_grid_token=False,
        use_textual_state_desc=False,
    )
    user_text += (
        " Output requirement: provide the final path ONLY in this exact form: "
        "<answer>go right -> go down -> ...</answer>. "
        "Use only actions from {go left, go right, go up, go down}."
    )
    image_path = sample["images"][0]
    data_url = image_path_to_data_url(image_path)
    system_text = GRID_SYSTEM_MESSAGE_TEXT_ONLY
    return system_text, user_text, data_url


def create_client(api_key: str, base_url: Optional[str]):
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is required. Install with: pip install openai") from exc
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _extract_text_from_chat_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
                elif isinstance(item.get("text"), str):
                    chunks.append(item["text"])
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunks).strip()
    return str(content)


def call_api_chat(client, model: str, system_text: str, user_text: str, image_data_url: str, max_output_tokens: int, temperature: float, timeout: float):
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_output_tokens,
        timeout=timeout,
        messages=[
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    )
    msg = resp.choices[0].message
    raw_text = _extract_text_from_chat_content(msg.content)
    usage = getattr(resp, "usage", None)
    return raw_text, usage


def call_api_responses(client, model: str, system_text: str, user_text: str, image_data_url: str, max_output_tokens: int, temperature: float, timeout: float):
    resp = client.responses.create(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_text}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_text},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            },
        ],
    )
    raw_text = getattr(resp, "output_text", None)
    if not raw_text:
        raw_text = str(resp)
    usage = getattr(resp, "usage", None)
    return raw_text, usage


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    if args.max_samples and args.max_samples > 0:
        dataset = dataset[: args.max_samples]

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env: {args.api_key_env}")
    client = create_client(api_key=api_key, base_url=args.api_base_url)

    records = []
    totals = {
        "num_samples": len(dataset),
        "num_reached_goal": 0,
        "num_exact_match": 0,
        "num_declared_no_path": 0,
        "num_no_answer": 0,
    }
    status_counts: Dict[str, int] = {}
    total_prompt_tokens = 0
    total_completion_tokens = 0

    started = time.time()

    pbar = tqdm(range(len(dataset)), total=len(dataset), desc="Evaluating API", unit="sample")
    for sample_index in pbar:
        sample = dataset[sample_index]
        reference = sample["conversations"][-1]["value"]
        system_text, user_text, data_url = build_messages(sample=sample, reference=reference)

        last_exc = None
        raw_output = ""
        usage = None
        for _ in range(max(args.max_retries, 1)):
            try:
                if args.api_mode == "responses":
                    raw_output, usage = call_api_responses(
                        client=client,
                        model=args.model,
                        system_text=system_text,
                        user_text=user_text,
                        image_data_url=data_url,
                        max_output_tokens=args.max_output_tokens,
                        temperature=args.temperature,
                        timeout=args.request_timeout,
                    )
                else:
                    raw_output, usage = call_api_chat(
                        client=client,
                        model=args.model,
                        system_text=system_text,
                        user_text=user_text,
                        image_data_url=data_url,
                        max_output_tokens=args.max_output_tokens,
                        temperature=args.temperature,
                        timeout=args.request_timeout,
                    )
                break
            except Exception as exc:
                last_exc = exc
                time.sleep(args.retry_sleep)
        else:
            raise RuntimeError(f"API request failed after retries: {last_exc}")

        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens) or 0
                completion_tokens = usage.get("completion_tokens", completion_tokens) or 0
            total_prompt_tokens += int(prompt_tokens)
            total_completion_tokens += int(completion_tokens)

        predicted_answer = extract_answer(raw_output)
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
                "image_path": sample["images"][0] if sample.get("images") else None,
            }
        )

        processed = sample_index + 1
        live_reached = totals["num_reached_goal"] / max(processed, 1)
        live_exact = totals["num_exact_match"] / max(processed, 1)
        pbar.set_postfix(
            processed=f"{processed}/{len(dataset)}",
            reached_acc=f"{live_reached:.3f}",
            exact_acc=f"{live_exact:.3f}",
        )

    elapsed = time.time() - started
    totals["reached_goal_accuracy"] = totals["num_reached_goal"] / max(totals["num_samples"], 1)
    totals["exact_match_accuracy"] = totals["num_exact_match"] / max(totals["num_samples"], 1)
    totals["status_counts"] = status_counts
    totals["elapsed_seconds"] = round(elapsed, 2)
    totals["prompt_tokens"] = total_prompt_tokens
    totals["completion_tokens"] = total_completion_tokens

    timestamp = datetime.now(timezone.utc).astimezone().isoformat()
    metadata = {
        "timestamp": timestamp,
        "api_mode": args.api_mode,
        "model": args.model,
        "api_base_url": args.api_base_url,
        "dataset_path": args.dataset_path,
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "max_retries": args.max_retries,
        "retry_sleep": args.retry_sleep,
        "request_timeout": args.request_timeout,
        "prompt_mode": "text_only",
        "note": "API baseline eval on provided dataset.",
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(totals, f, ensure_ascii=False, indent=2)
    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    summary_lines = [
        "API Baseline Eval Summary",
        f"DateTime: {timestamp}",
        f"Model: {args.model}",
        f"APIMode: {args.api_mode}",
        f"APIBaseURL: {args.api_base_url}",
        f"Dataset: {args.dataset_path}",
        f"Samples: {totals['num_samples']}",
        f"ReachedGoalAccuracy: {totals['reached_goal_accuracy']:.4f}",
        f"ExactMatchAccuracy: {totals['exact_match_accuracy']:.4f}",
        f"ReachedGoalCount: {totals['num_reached_goal']}",
        f"ExactMatchCount: {totals['num_exact_match']}",
        f"NoAnswerCount: {totals['num_no_answer']}",
        f"DeclaredNoPathCount: {totals['num_declared_no_path']}",
        f"ElapsedSeconds: {totals['elapsed_seconds']}",
        f"PromptTokens: {totals['prompt_tokens']}",
        f"CompletionTokens: {totals['completion_tokens']}",
        f"StatusCounts: {json.dumps(status_counts, ensure_ascii=False)}",
        "",
        metadata["note"],
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"saved to {output_dir}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
