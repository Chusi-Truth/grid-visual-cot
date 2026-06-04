#!/usr/bin/env python3
"""Recompute eval metrics from saved results with FrozenLake boundary semantics."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


MOVE_PATTERN = re.compile(r"go\s+(left|right|up|down)", re.IGNORECASE)
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
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--threshold-area", type=int, default=20)
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


def simulate_answer(reference_text, predicted_answer):
    start = parse_start(reference_text)
    goal = parse_goal(reference_text)
    if goal is None:
        raise ValueError("Could not parse goal coordinates from reference text")
    holes = parse_holes(reference_text)
    rows, cols = infer_bounds(start, goal, holes)

    if predicted_answer is None:
        return {"status": "no_answer", "trajectory": [start]}

    if predicted_answer.strip().lower() == "fail to find a valid path":
        return {"status": "declared_no_path", "trajectory": [start]}

    moves = [m.lower() for m in MOVE_PATTERN.findall(predicted_answer)]
    position = start
    trajectory = [position]

    for move in moves:
        next_pos = apply_move(position, move)
        # FrozenLake: out-of-bound actions keep position unchanged.
        if not (0 <= next_pos[0] < rows and 0 <= next_pos[1] < cols):
            next_pos = position
        trajectory.append(next_pos)
        if next_pos in holes:
            return {"status": "fell_into_hole", "trajectory": trajectory}
        position = next_pos
        if position == goal:
            return {"status": "reached_goal", "trajectory": trajectory}

    return {"status": "stopped_before_goal", "trajectory": trajectory}


def bucket_name(rows, cols, threshold):
    area = rows * cols
    if area < threshold:
        return f"lt_{threshold}"
    if area > threshold:
        return f"gt_{threshold}"
    return f"eq_{threshold}"


def main():
    args = parse_args()
    dataset = json.load(open(args.dataset_path, encoding="utf-8"))
    records = json.load(open(args.results_path, encoding="utf-8"))

    status_counts = defaultdict(int)
    bucket_counts = defaultdict(lambda: {"num_samples": 0, "num_reached_goal": 0, "status_counts": defaultdict(int)})
    num_reached = 0

    for rec in records:
        idx = rec["sample_index"]
        ref_text = dataset[idx]["conversations"][-1]["value"]
        sim = simulate_answer(ref_text, rec.get("predicted_answer"))
        rec["simulation_recomputed"] = sim
        rec["reached_goal_recomputed"] = sim["status"] == "reached_goal"

        rows = rec["rows"]
        cols = rec["cols"]
        b = bucket_name(rows, cols, args.threshold_area)
        bucket_counts[b]["num_samples"] += 1
        bucket_counts[b]["status_counts"][sim["status"]] += 1
        if sim["status"] == "reached_goal":
            bucket_counts[b]["num_reached_goal"] += 1
            num_reached += 1
        status_counts[sim["status"]] += 1

    total = len(records)
    summary = {
        "num_samples": total,
        "num_reached_goal": num_reached,
        "reached_goal_accuracy": num_reached / max(total, 1),
        "status_counts": dict(status_counts),
        "by_bucket": {},
    }
    for b, v in bucket_counts.items():
        summary["by_bucket"][b] = {
            "num_samples": v["num_samples"],
            "num_reached_goal": v["num_reached_goal"],
            "reached_goal_accuracy": v["num_reached_goal"] / max(v["num_samples"], 1),
            "status_counts": dict(v["status_counts"]),
        }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
