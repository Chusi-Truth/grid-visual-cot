#!/usr/bin/env python3
"""Build reachable holdout set using map-level graph search (BFS/DFS), not teacher answer."""

import argparse
import json
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


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
    parser.add_argument("--input-candidates", required=True, help="Candidate dataset json path")
    parser.add_argument("--output-dataset", required=True, help="Output reachable dataset json path")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--search", choices=["bfs", "dfs"], default="bfs")
    return parser.parse_args()


def parse_start(text):
    m = START_PATTERN.search(text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return (0, 0)


def parse_goal(text):
    m = GOAL_PATTERN.search(text)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = FROM_TO_PATTERN.search(text)
    if m:
        return int(m.group(3)), int(m.group(4))
    return None


def parse_holes(text):
    m = HOLES_LINE_PATTERN.search(text)
    if not m:
        return set()
    return {tuple(map(int, xy)) for xy in COORD_PATTERN.findall(m.group(1))}


def infer_bounds(start, goal, holes):
    coords = [start, goal, *holes]
    max_row = max(r for r, _ in coords)
    max_col = max(c for _, c in coords)
    return max_row + 1, max_col + 1


def neighbors(r, c, rows, cols):
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def is_reachable(start, goal, holes, rows, cols, search):
    if start in holes or goal in holes:
        return False
    if start == goal:
        return True

    seen = {start}
    if search == "bfs":
        q = deque([start])
        pop = q.popleft
        push = q.append
    else:
        q = [start]
        pop = q.pop
        push = q.append

    while q:
        cur = pop()
        for nxt in neighbors(cur[0], cur[1], rows, cols):
            if nxt in holes or nxt in seen:
                continue
            if nxt == goal:
                return True
            seen.add(nxt)
            push(nxt)
    return False


def main():
    args = parse_args()
    input_path = Path(args.input_candidates)
    output_path = Path(args.output_dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open(encoding="utf-8") as f:
        candidates = json.load(f)

    selected = []
    skipped_parse = 0
    skipped_unreachable = 0

    for sample in candidates:
        if len(selected) >= args.num_samples:
            break
        reference = sample["conversations"][-1]["value"]
        start = parse_start(reference)
        goal = parse_goal(reference)
        holes = parse_holes(reference)
        if goal is None:
            skipped_parse += 1
            continue
        rows, cols = infer_bounds(start, goal, holes)
        ok = is_reachable(start, goal, holes, rows, cols, args.search)
        if not ok:
            skipped_unreachable += 1
            continue
        selected.append(sample)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    metadata = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "input_candidates": str(input_path),
        "output_dataset": str(output_path),
        "selection_rule": f"first {args.num_samples} samples whose map is reachable by {args.search.upper()} graph search (ignoring teacher answer)",
        "num_samples": len(selected),
        "target_num_samples": args.num_samples,
        "search": args.search,
        "skipped_parse": skipped_parse,
        "skipped_unreachable": skipped_unreachable,
    }
    meta_path = output_path.parent / "metadata_search_reachable.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Reachable Holdout (Graph Search)")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Samples: {len(selected)} / {args.num_samples}")
    print(f"Search: {args.search}")
    print(f"SkippedParse: {skipped_parse}")
    print(f"SkippedUnreachable: {skipped_unreachable}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()

