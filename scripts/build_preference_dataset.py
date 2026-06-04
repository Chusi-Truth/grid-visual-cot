#!/usr/bin/env python3
"""Build a simple chosen/rejected preference dataset from existing SFT data.

The chosen response is the original successful assistant response.
The rejected response is synthesized by perturbing the final answer path while
keeping the rest of the response intact. We simulate the perturbed path and
assign a scalar reward based on the resulting outcome.
"""

import argparse
import json
import random
import re
from pathlib import Path


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

MOVE_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ALL_MOVES = ["up", "down", "left", "right"]
REWARD_BY_STATUS = {
    "reached_goal": 1.0,
    "stopped_before_goal": 0.2,
    "declared_no_path": 0.0,
    "no_answer": -0.1,
    "fell_into_hole": -0.2,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
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
    raise ValueError("Could not parse goal coordinates")


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


def extract_moves(text):
    return [m.lower() for m in MOVE_PATTERN.findall(text or "")]


def replace_answer_block(response_text, answer_text):
    answer_block = f"<answer>{answer_text}</answer>"
    if ANSWER_PATTERN.search(response_text):
        return ANSWER_PATTERN.sub(answer_block, response_text, count=1)
    return response_text.rstrip() + "\n" + answer_block


def format_path_answer(moves):
    if not moves:
        return "Final chosen trace: fail to find a valid path"
    path = "->".join(f"go {move}" for move in moves)
    return f"Final chosen trace: {path}"


def apply_move(position, move):
    dr, dc = MOVE_TO_DELTA[move]
    return position[0] + dr, position[1] + dc


def simulate_answer(reference_text, answer_text):
    start = parse_start(reference_text)
    goal = parse_goal(reference_text)
    holes = parse_holes(reference_text)
    rows, cols = infer_bounds(start, goal, holes)

    if answer_text is None:
        return {"status": "no_answer", "trajectory": [start]}

    lower = answer_text.strip().lower()
    if "fail to find a valid path" in lower:
        return {"status": "declared_no_path", "trajectory": [start]}

    moves = extract_moves(answer_text)
    if not moves:
        return {"status": "no_answer", "trajectory": [start]}

    position = start
    trajectory = [position]
    for move in moves:
        next_pos = apply_move(position, move)
        if not (0 <= next_pos[0] < rows and 0 <= next_pos[1] < cols):
            next_pos = position
        trajectory.append(next_pos)
        if next_pos in holes:
            return {"status": "fell_into_hole", "trajectory": trajectory}
        position = next_pos
        if position == goal:
            return {"status": "reached_goal", "trajectory": trajectory}

    return {"status": "stopped_before_goal", "trajectory": trajectory}


def generate_negative_candidates(reference_text, gold_moves, rng):
    candidates = []

    if gold_moves:
        truncated = gold_moves[:-1]
        candidates.append(("truncate_last", format_path_answer(truncated)))

    if gold_moves:
        last = gold_moves[-1]
        for alt in ALL_MOVES:
            if alt != last:
                mutated = list(gold_moves)
                mutated[-1] = alt
                candidates.append((f"flip_last_to_{alt}", format_path_answer(mutated)))

    if len(gold_moves) >= 2:
        prefix_len = rng.randint(1, len(gold_moves) - 1)
        prefix = gold_moves[:prefix_len]
        alt = rng.choice(ALL_MOVES)
        candidates.append(("random_prefix", format_path_answer(prefix + [alt])))

    candidates.append(("declare_no_path", "Final chosen trace: fail to find a valid path"))
    candidates.append(("empty_answer", ""))

    dedup = []
    seen = set()
    for source, answer in candidates:
        key = answer.strip()
        if key in seen:
            continue
        seen.add(key)
        dedup.append((source, answer))
    return dedup


def choose_rejected(reference_text, response_text, rng):
    match = ANSWER_PATTERN.search(response_text)
    if not match:
        raise ValueError("Response is missing an <answer>...</answer> block")

    gold_answer = match.group(1).strip()
    gold_moves = extract_moves(gold_answer)
    if not gold_moves:
        raise ValueError("Gold answer does not contain any moves")

    candidates = []
    for source, candidate_answer in generate_negative_candidates(reference_text, gold_moves, rng):
        sim = simulate_answer(reference_text, candidate_answer if candidate_answer else None)
        if sim["status"] == "reached_goal":
            continue
        reward = REWARD_BY_STATUS[sim["status"]]
        candidates.append(
            {
                "rejected_response": replace_answer_block(response_text, candidate_answer),
                "rejected_answer": candidate_answer if candidate_answer else None,
                "rejected_status": sim["status"],
                "rejected_reward": reward,
                "negative_source": source,
            }
        )

    if not candidates:
        fallback_answer = "Final chosen trace: fail to find a valid path"
        return {
            "rejected_response": replace_answer_block(response_text, fallback_answer),
            "rejected_answer": fallback_answer,
            "rejected_status": "declared_no_path",
            "rejected_reward": REWARD_BY_STATUS["declared_no_path"],
            "negative_source": "fallback_no_path",
        }

    by_status = {}
    for item in candidates:
        by_status.setdefault(item["rejected_status"], []).append(item)

    # Prefer harder negatives, but keep some diversity so the model also learns
    # to reject hole-fall / no-answer / declared-no-path cases.
    status_priority = [
        ("stopped_before_goal", 0.55),
        ("fell_into_hole", 0.25),
        ("declared_no_path", 0.15),
        ("no_answer", 0.05),
    ]
    draw = rng.random()
    cumulative = 0.0
    chosen_status = None
    for status, prob in status_priority:
        cumulative += prob
        if draw <= cumulative and status in by_status:
            chosen_status = status
            break
    if chosen_status is None:
        chosen_status = max(by_status, key=lambda status: REWARD_BY_STATUS[status])

    pool = by_status[chosen_status]
    pool.sort(key=lambda item: (item["rejected_reward"], item["negative_source"]), reverse=True)
    shortlist = pool[: min(3, len(pool))]
    return rng.choice(shortlist)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.max_samples > 0:
        data = data[: args.max_samples]

    output = []
    skipped = 0

    for idx, sample in enumerate(data):
        conversations = sample.get("conversations", [])
        if len(conversations) < 2:
            skipped += 1
            continue

        assistant = conversations[-1]["value"]
        try:
            rejected = choose_rejected(assistant, assistant, rng)
        except Exception:
            skipped += 1
            continue

        output.append(
            {
                "source_index": idx,
                "images": sample.get("images"),
                "conversations": sample["conversations"],
                "chosen_response": assistant,
                "chosen_reward": 1.0,
                **rejected,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "num_samples": len(output),
                "skipped": skipped,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
