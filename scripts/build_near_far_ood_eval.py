#!/usr/bin/env python3
"""Build strict near/far OOD FrozenLake evaluation sets."""

import argparse
import json
import sys
from collections import Counter, deque
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
    parser.add_argument("--num-near", type=int, default=100)
    parser.add_argument("--num-far", type=int, default=100)
    parser.add_argument("--near-sizes", default="4x6,6x4,4x7,6x5")
    parser.add_argument("--far-sizes", default="6x6,6x7,7x6,7x7")
    parser.add_argument("--p", type=float, default=0.8)
    parser.add_argument("--base-seed-near", type=int, default=810000)
    parser.add_argument("--base-seed-far", type=int, default=910000)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def parse_sizes(spec):
    out = []
    for part in spec.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "x" not in part:
            raise ValueError(f"Invalid size: {part}")
        r, c = part.split("x", 1)
        out.append((int(r), int(c)))
    if not out:
        raise ValueError("Size list is empty")
    return out


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
    if not holes:
        hole_line = "There are no holes I must avoid."
    elif len(holes) == 1:
        hole_line = f"There is 1 hole I must avoid: ({holes[0][0]},{holes[0][1]})."
    else:
        hole_str = ", ".join(f"({r},{c})" for r, c in holes)
        hole_line = f"There are {len(holes)} holes I must avoid: {hole_str}."
    return "\n".join(
        [
            "<think>",
            f"This is a FrozenLake pathfinding problem on a {rows}x{cols} grid.",
            f"I start at (0, 0) and the goal is at ({goal[0]}, {goal[1]}).",
            hole_line,
            "I need to avoid holes and produce a valid path from start to goal.",
            "</think>",
            f"<answer>{answer_text}</answer>",
        ]
    )


def build_one_split(split_name, output_root, sizes, num_samples, p, base_seed, image_size):
    split_dir = output_root / split_name
    image_dir = split_dir / "test_images"
    split_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    size_dist = Counter()
    seed_cursor = base_seed
    sample_id = 0

    while len(samples) < num_samples:
        rows, cols = sizes[len(samples) % len(sizes)]
        desc = generate_map_from_seed(seed_cursor, rows=rows, cols=cols, p=p)
        path = bfs_shortest_path(desc)
        if path is None:
            seed_cursor += 1
            continue

        image_path = image_dir / f"{sample_id:04d}_{rows}x{cols}_seed{seed_cursor}.jpg"
        render_image(desc, image_path, image_size)
        answer_text = " -> ".join(ACTION_TO_TEXT[a] for a in path)
        holes = hole_positions(desc)
        ref = build_reference_text(rows, cols, holes, answer_text)

        samples.append(
            {
                "conversations": [
                    {"from": "human", "value": "<image>"},
                    {"from": "gpt", "value": ref},
                ],
                "images": [str(image_path)],
                "meta": {
                    "split": split_name,
                    "rows": rows,
                    "cols": cols,
                    "area": rows * cols,
                    "seed": seed_cursor,
                },
            }
        )
        size_dist[f"{rows}x{cols}"] += 1
        seed_cursor += 1
        sample_id += 1

    dataset_path = split_dir / f"grid_cot_test_{split_name}.json"
    with dataset_path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    metadata = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "split": split_name,
        "dataset_path": str(dataset_path),
        "num_samples": len(samples),
        "sizes": [f"{r}x{c}" for r, c in sizes],
        "size_distribution": dict(size_dist),
        "generation_args": {
            "p": p,
            "base_seed": base_seed,
            "image_size": image_size,
        },
    }
    with (split_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    summary_lines = [
        f"{split_name} OOD Eval Set",
        f"Dataset: {dataset_path}",
        f"Samples: {len(samples)}",
        f"Sizes: {', '.join(metadata['sizes'])}",
        f"SizeDistribution: {json.dumps(dict(size_dist), ensure_ascii=False, sort_keys=True)}",
    ]
    (split_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "split": split_name,
        "dataset_path": str(dataset_path),
        "num_samples": len(samples),
        "size_distribution": dict(size_dist),
    }


def main():
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    near_sizes = parse_sizes(args.near_sizes)
    far_sizes = parse_sizes(args.far_sizes)

    near_info = build_one_split(
        split_name="near_ood",
        output_root=output_root,
        sizes=near_sizes,
        num_samples=args.num_near,
        p=args.p,
        base_seed=args.base_seed_near,
        image_size=args.image_size,
    )
    far_info = build_one_split(
        split_name="far_ood",
        output_root=output_root,
        sizes=far_sizes,
        num_samples=args.num_far,
        p=args.p,
        base_seed=args.base_seed_far,
        image_size=args.image_size,
    )

    top_meta = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "output_dir": str(output_root),
        "near_ood": near_info,
        "far_ood": far_info,
    }
    with (output_root / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(top_meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(top_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
