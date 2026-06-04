#!/usr/bin/env python3
"""Build a reachable mixed-size FrozenLake evaluation set."""

import argparse
import json
import os
import sys
from collections import Counter, deque, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
COTCREATOR_BAGEL_DIR = Path("/root/CoTCreator/eval/bagel_plan")
sys.path.insert(0, str(COTCREATOR_BAGEL_DIR))

from flcreator import generate_map_from_seed  # type: ignore


ACTION_TO_TEXT = {
    0: "go left",
    1: "go down",
    2: "go right",
    3: "go up",
}

MOVE_DIRS = [
    (0, -1, 0),
    (1, 0, 1),
    (0, 1, 2),
    (-1, 0, 3),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-small", type=int, default=50, help="area < threshold_area")
    parser.add_argument("--num-large", type=int, default=50, help="area > threshold_area")
    parser.add_argument("--threshold-area", type=int, default=20)
    parser.add_argument("--min-size", type=int, default=4)
    parser.add_argument("--max-size", type=int, default=7)
    parser.add_argument("--p", type=float, default=0.8)
    parser.add_argument("--base-seed", type=int, default=700000)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def bucket_for_size(rows, cols, threshold_area):
    area = rows * cols
    if area < threshold_area:
        return "small"
    if area > threshold_area:
        return "large"
    return "equal"


def enumerate_sizes(min_size, max_size, threshold_area):
    buckets = defaultdict(list)
    for rows in range(min_size, max_size + 1):
        for cols in range(min_size, max_size + 1):
            bucket = bucket_for_size(rows, cols, threshold_area)
            buckets[bucket].append((rows, cols))
    return buckets


def bfs_shortest_path(desc):
    rows = len(desc)
    cols = len(desc[0])
    start = (0, 0)
    goal = (rows - 1, cols - 1)
    q = deque([(start, [])])
    seen = {start}

    while q:
        (r, c), path = q.popleft()
        if (r, c) == goal:
            return path
        for dr, dc, action in MOVE_DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if (nr, nc) in seen:
                continue
            if desc[nr][nc] == "H":
                continue
            seen.add((nr, nc))
            q.append(((nr, nc), path + [action]))
    return None


def hole_positions(desc):
    holes = []
    for r, row in enumerate(desc):
        for c, cell in enumerate(row):
            if cell == "H":
                holes.append((r, c))
    return holes


def render_image(desc, image_path, image_size):
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        is_slippery=False,
        render_mode="rgb_array",
    )
    env.reset(seed=0)
    frame = env.render()
    env.close()

    img = Image.fromarray(frame).convert("RGB")
    img = img.resize((image_size, image_size), resample=Image.NEAREST)
    img.save(image_path, format="JPEG", quality=95, subsampling=0)


def build_reference_text(rows, cols, holes, answer_text):
    goal = (rows - 1, cols - 1)
    hole_count = len(holes)
    if hole_count == 0:
        hole_line = "There are no holes I must avoid."
    elif hole_count == 1:
        hole_line = f"There is 1 hole I must avoid: ({holes[0][0]},{holes[0][1]})."
    else:
        holes_str = ", ".join(f"({r},{c})" for r, c in holes)
        hole_line = f"There are {hole_count} holes I must avoid: {holes_str}."

    think = "\n".join(
        [
            "<think>",
            f"This is a FrozenLake pathfinding problem on a {rows}x{cols} grid.",
            f"I start at (0, 0) and the goal is at ({goal[0]}, {goal[1]}).",
            hole_line,
            "I need to avoid holes and produce a valid path from start to goal.",
            "</think>",
        ]
    )
    answer = f"<answer>{answer_text}</answer>"
    return think + "\n" + answer


def make_sample(image_path, rows, cols, holes, answer_text, seed):
    return {
        "conversations": [
            {"from": "human", "value": "<image>"},
            {"from": "gpt", "value": build_reference_text(rows, cols, holes, answer_text)},
        ],
        "images": [str(image_path)],
        "meta": {
            "rows": rows,
            "cols": cols,
            "area": rows * cols,
            "bucket": bucket_for_size(rows, cols, 20),
            "seed": seed,
        },
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "test_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    size_buckets = enumerate_sizes(args.min_size, args.max_size, args.threshold_area)
    if not size_buckets["small"]:
        raise ValueError("No size combinations with area smaller than threshold_area")
    if not size_buckets["large"]:
        raise ValueError("No size combinations with area larger than threshold_area")

    targets = {"small": args.num_small, "large": args.num_large}
    counters = Counter()
    size_usage = Counter()
    samples = []
    seed_cursor = args.base_seed
    sample_id = 0

    while counters["small"] < targets["small"] or counters["large"] < targets["large"]:
        for bucket in ("small", "large"):
            if counters[bucket] >= targets[bucket]:
                continue

            size_list = size_buckets[bucket]
            size_idx = counters[bucket] % len(size_list)
            rows, cols = size_list[size_idx]
            desc = generate_map_from_seed(seed_cursor, rows=rows, cols=cols, p=args.p)
            path = bfs_shortest_path(desc)
            if path is None:
                seed_cursor += 1
                continue

            image_path = image_dir / f"{sample_id:04d}_{rows}x{cols}_seed{seed_cursor}.jpg"
            render_image(desc, image_path, args.image_size)
            answer_text = " -> ".join(ACTION_TO_TEXT[a] for a in path)
            holes = hole_positions(desc)
            sample = make_sample(image_path, rows, cols, holes, answer_text, seed_cursor)
            sample["meta"]["bucket"] = bucket
            samples.append(sample)

            counters[bucket] += 1
            size_usage[f"{rows}x{cols}"] += 1
            seed_cursor += 1
            sample_id += 1

    dataset_path = output_dir / "grid_cot_test_mixed_size_reachable.json"
    with dataset_path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    metadata = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "dataset_path": str(dataset_path),
        "num_samples": len(samples),
        "threshold_area": args.threshold_area,
        "bucket_counts": dict(counters),
        "size_distribution": dict(size_usage),
        "policy": {
            "small": f"rows*cols < {args.threshold_area}",
            "large": f"rows*cols > {args.threshold_area}",
            "equal": f"rows*cols == {args.threshold_area} is excluded from this dataset",
        },
        "generation_args": {
            "min_size": args.min_size,
            "max_size": args.max_size,
            "p": args.p,
            "base_seed": args.base_seed,
            "image_size": args.image_size,
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    summary_lines = [
        "Mixed-Size Reachable Test Set",
        f"Dataset: {dataset_path}",
        f"Samples: {len(samples)}",
        f"SmallCount(area<{args.threshold_area}): {counters['small']}",
        f"LargeCount(area>{args.threshold_area}): {counters['large']}",
        f"ExcludedEqualArea: {args.threshold_area}",
        f"SizeDistribution: {json.dumps(dict(size_usage), ensure_ascii=False, sort_keys=True)}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
