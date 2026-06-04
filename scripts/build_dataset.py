#!/usr/bin/env python3
"""
Build grid CoT dataset with dense visual checkpoints.

Teacher search algorithm:
  - Default: BFS (deterministic, reachable map => guaranteed path)
  - Optional: DFS
  - Optional legacy: commit-based beam search

Usage:
  python scripts/build_grid_cot_dataset.py \
      --sample_num 500 \
      --checkpoint_interval 2 \
      --output_json /home/hanxujie/data/grid_cot_dense.json \
      --checkpoint_dir /home/hanxujie/data/grid_checkpoints
"""

import argparse
import json
import os
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

# Headless rendering setup (must precede gymnasium import)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")

import gymnasium as gym
from PIL import Image

def resolve_bagel_plan_dir():
    candidates = []
    env_path = os.environ.get("COTCREATOR_BAGEL_PLAN_DIR")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path("/root/CoTCreator/eval/bagel_plan"),
            Path("/home/hanxujie/CoTCreator/eval/bagel_plan"),
        ]
    )

    for candidate in candidates:
        if (candidate / "flcreator.py").exists() and (candidate / "beamSearch.py").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate CoTCreator bagel_plan directory. "
        "Set COTCREATOR_BAGEL_PLAN_DIR to the directory containing flcreator.py and beamSearch.py."
    )


BAGEL_PLAN_DIR = resolve_bagel_plan_dir()
BAGEL_OUTPUT_DIR = BAGEL_PLAN_DIR / "cot_visual_500"


# ---------------------------------------------------------------------------
#  Import utilities from CoTCreator
# ---------------------------------------------------------------------------
sys.path.insert(0, str(BAGEL_PLAN_DIR))
from flcreator import SeedFrozenLake, set_global_seed
from beamSearch import (
    pre_definition,
    step_cot,
    getHolePos,
    toCotTrcace,
    _is_reachable,
    _has_free_neighbor,
)


# ---------------------------------------------------------------------------
#  Checkpoint helpers
# ---------------------------------------------------------------------------

def get_adjacent_holes(pos, holes, rows, cols):
    """Return hole positions adjacent to *pos*."""
    r, c = pos
    hole_set = set(holes)
    adjacent = []
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) in hole_set:
            adjacent.append((nr, nc))
    return adjacent


def render_checkpoint(desc, trace, seed, image_size=256):
    """Render the grid after executing *trace*, returning a PIL Image."""
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        is_slippery=False,
        render_mode="rgb_array",
    )
    env.reset(seed=seed)
    for action in trace:
        env.step(action)
    frame = env.render()
    env.close()
    img = Image.fromarray(frame).convert("RGB")
    img = img.resize((image_size, image_size), resample=Image.NEAREST)
    return img


def ensure_base_image(desc, image_path, seed, image_size=256):
    """Render and save the base grid image if it does not already exist."""
    image_path = Path(image_path)
    if image_path.exists():
        return
    image_path.parent.mkdir(parents=True, exist_ok=True)
    img = render_checkpoint(desc, [], seed, image_size=image_size)
    img.save(image_path)


def visual_check_text(pos, goal, holes, rows, cols, checkpoint_idx, sample_idx):
    """Generate varied visual-check description text."""
    r, c = pos
    dist = abs(goal[0] - r) + abs(goal[1] - c)
    adjacent = get_adjacent_holes(pos, holes, rows, cols)

    dash = "\u2014"
    if adjacent:
        hole_str = str(adjacent)
        warnings = [
            f"Watch out {dash} hole(s) adjacent at {hole_str}.",
            f"Careful {dash} nearby hole(s) at {hole_str}.",
            f"Note: hole(s) next to me at {hole_str}.",
        ]
        warning = warnings[(checkpoint_idx + sample_idx) % len(warnings)]
    else:
        clears = [
            "Path ahead looks clear.",
            f"No holes nearby {dash} safe to continue.",
            "Surroundings are clear.",
        ]
        warning = clears[(checkpoint_idx + sample_idx) % len(clears)]

    templates = [
        f"(Visual check: I'm at ({r},{c}), {dist} step(s) from goal. {warning} Continuing.)",
        f"(Checking the grid: position ({r},{c}), distance to goal = {dist}. {warning} Moving on.)",
        f"(Grid snapshot: currently at ({r},{c}), {dist} step(s) left. {warning} Proceeding.)",
    ]
    return templates[(checkpoint_idx + sample_idx) % len(templates)]


# ---------------------------------------------------------------------------
#  Search helpers (BFS / DFS)
# ---------------------------------------------------------------------------

ACTION_DELTAS = {
    0: (0, -1),  # left
    1: (1, 0),   # down
    2: (0, 1),   # right
    3: (-1, 0),  # up
}

# Prefer down/right first so traces tend to move toward bottom-right goal.
SEARCH_ACTION_ORDER = (1, 2, 0, 3)


def normalize_desc(desc):
    """Convert env desc to a 2D char grid."""
    grid = []
    for row in desc:
        if isinstance(row, bytes):
            row = row.decode("utf-8")
        elif not isinstance(row, str):
            row = "".join(row)
        grid.append(list(row))
    return grid


def find_start_goal(grid):
    start = None
    goal = None
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == "S":
                start = (r, c)
            elif cell == "G":
                goal = (r, c)

    if start is None:
        start = (0, 0)
    if goal is None:
        goal = (len(grid) - 1, len(grid[0]) - 1)
    return start, goal


def next_pos(pos, action, rows, cols):
    dr, dc = ACTION_DELTAS[action]
    nr = min(max(pos[0] + dr, 0), rows - 1)
    nc = min(max(pos[1] + dc, 0), cols - 1)
    return nr, nc


def solve_path_trace(desc, algo="bfs"):
    """Return a safe action trace to goal, or None if not found."""
    grid = normalize_desc(desc)
    rows, cols = len(grid), len(grid[0])
    start, goal = find_start_goal(grid)
    holes = {
        (r, c)
        for r, row in enumerate(grid)
        for c, cell in enumerate(row)
        if cell == "H"
    }

    if start in holes or goal in holes:
        return None
    if start == goal:
        return []

    if algo == "bfs":
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            pos, trace = queue.popleft()
            for action in SEARCH_ACTION_ORDER:
                nxt = next_pos(pos, action, rows, cols)
                if nxt in holes or nxt in visited:
                    continue
                new_trace = trace + [action]
                if nxt == goal:
                    return new_trace
                visited.add(nxt)
                queue.append((nxt, new_trace))
        return None

    if algo == "dfs":
        stack = [(start, [])]
        visited = {start}
        while stack:
            pos, trace = stack.pop()
            if pos == goal:
                return trace
            for action in reversed(SEARCH_ACTION_ORDER):
                nxt = next_pos(pos, action, rows, cols)
                if nxt in holes or nxt in visited:
                    continue
                visited.add(nxt)
                stack.append((nxt, trace + [action]))
        return None

    raise ValueError(f"Unsupported search algorithm: {algo}")


def evaluate_trace(seed_env, episode_used, trace, goal_pos):
    """Rollout one trace and return fields used by step_cot."""
    seed_env.reset(episode_idx=episode_used)
    for action in trace:
        seed_env.step(action)
    return {
        "trace": trace,
        "agent_pos": seed_env.getAgentPos(),
        "goal_pos": goal_pos,
        "reward": seed_env.reward(),
    }


# ---------------------------------------------------------------------------
#  Core dataset builder
# ---------------------------------------------------------------------------

def build_dataset(args):
    set_global_seed(0)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seed_env = SeedFrozenLake(
        rows=args.grid_rows,
        cols=args.grid_cols,
        p=args.map_p,
        is_slippery=False,
        base_seed=args.map_seed,
        render_mode=None,
    )

    json_all = []
    next_episode = 0
    stats = {"total": 0, "success": 0, "fail": 0}

    for idx in range(args.sample_num):
        # --- Find a reachable map (same logic as beamSearch.py) ---
        desc = None
        episode_used = next_episode
        for extra in range(50):
            seed_env.reset(episode_idx=next_episode + extra)
            desc = seed_env.get_map_desc()
            if _has_free_neighbor(desc) and _is_reachable(desc):
                episode_used = next_episode + extra
                break
        else:
            next_episode += 50
            continue

        next_episode = episode_used + 1
        image_idx = args.start_idx + episode_used
        map_seed = args.map_seed + episode_used

        holes = getHolePos(desc)
        seed_env.reset(episode_idx=episode_used)
        goal_pos = seed_env.getGoalPos()

        # --- CoT prefix ---
        cot_prefix = pre_definition(holes, goal_pos=goal_pos, sample_idx=idx)

        # --- Search + CoT rollout ---
        committed_trace = []
        all_cot_lines = [cot_prefix]
        checkpoint_images = []
        step_counter = 0
        checkpoint_idx = 0
        reached_goal = False

        if args.search_algo == "beam":
            while not reached_goal and step_counter < args.max_steps:
                trace_list = [committed_trace]
                ranked = None

                for _ in range(args.checkpoint_interval):
                    if step_counter >= args.max_steps:
                        break

                    # Expand candidates
                    trace_candidates = []
                    for trace in trace_list:
                        actions = random.sample(range(4), min(args.candidate_num, 4))
                        trace_candidates.extend([trace + [a] for a in actions])

                    # Evaluate each candidate
                    results = [
                        evaluate_trace(seed_env, episode_used, trace, goal_pos)
                        for trace in trace_candidates
                    ]

                    # Generate CoT text for this step
                    all_cot_lines.append(step_cot(step_counter, results, sample_idx=idx))
                    step_counter += 1

                    # Rank and keep top beams
                    ranked = sorted(results, key=lambda x: x["reward"], reverse=True)
                    trace_list = [item["trace"] for item in ranked[: args.beam_num]]

                    # Goal reached?
                    if ranked[0]["reward"] == 0:
                        reached_goal = True
                        committed_trace = ranked[0]["trace"]
                        break

                if reached_goal:
                    break

                if ranked is None:
                    break

                # --- COMMIT: take the best trace, discard the rest ---
                committed_trace = ranked[0]["trace"]
                agent_pos = ranked[0]["agent_pos"]

                # If committed position is a hole, abort — no useful checkpoint
                if ranked[0]["reward"] == -100:
                    break

                # Render checkpoint image
                ckpt_filename = f"{image_idx}_step{step_counter}.png"
                ckpt_path = checkpoint_dir / ckpt_filename
                img = render_checkpoint(desc, committed_trace, map_seed, args.image_size)
                img.save(ckpt_path)
                checkpoint_images.append(str(ckpt_path))

                # Insert <grid_token> + visual check text
                check = visual_check_text(
                    agent_pos,
                    goal_pos,
                    holes,
                    args.grid_rows,
                    args.grid_cols,
                    checkpoint_idx,
                    idx,
                )
                all_cot_lines.append(f"<grid_token>\n{check}")
                checkpoint_idx += 1
        else:
            solved_trace = solve_path_trace(desc, algo=args.search_algo)
            if solved_trace is not None:
                reached_goal = True
                for action in solved_trace:
                    # One-step local alternatives for CoT text.
                    trace_candidates = [committed_trace + [a] for a in range(4)]
                    results = [
                        evaluate_trace(seed_env, episode_used, trace, goal_pos)
                        for trace in trace_candidates
                    ]
                    all_cot_lines.append(step_cot(step_counter, results, sample_idx=idx))

                    committed_trace.append(action)
                    step_counter += 1

                    # Render dense checkpoints at configured interval.
                    if (
                        step_counter % args.checkpoint_interval == 0
                        or step_counter == len(solved_trace)
                    ):
                        state = evaluate_trace(seed_env, episode_used, committed_trace, goal_pos)
                        if state["reward"] == -100:
                            reached_goal = False
                            break
                        ckpt_filename = f"{image_idx}_step{step_counter}.png"
                        ckpt_path = checkpoint_dir / ckpt_filename
                        img = render_checkpoint(desc, committed_trace, map_seed, args.image_size)
                        img.save(ckpt_path)
                        checkpoint_images.append(str(ckpt_path))
                        check = visual_check_text(
                            state["agent_pos"],
                            goal_pos,
                            holes,
                            args.grid_rows,
                            args.grid_cols,
                            checkpoint_idx,
                            idx,
                        )
                        all_cot_lines.append(f"<grid_token>\n{check}")
                        checkpoint_idx += 1

        # --- Build final answer ---
        if reached_goal:
            final_answer = f"Final chosen trace: {toCotTrcace(committed_trace)}"
            stats["success"] += 1
        else:
            if not args.keep_failed:
                continue
            final_answer = "fail to find a valid path"
            stats["fail"] += 1
        stats["total"] += 1

        cot_full = "\n".join([
            "<think>",
            *all_cot_lines,
            "</think>",
            f"<answer>{final_answer}</answer>",
        ])

        # --- JSON entry ---
        base_image = f"{args.image_dir}/{image_idx}.jpg"
        ensure_base_image(desc, base_image, map_seed, image_size=args.image_size)
        images = [base_image] + checkpoint_images

        json_all.append({
            "conversations": [
                {"from": "human", "value": "<image>"},
                {"from": "gpt", "value": cot_full},
            ],
            "images": images,
        })

        if (idx + 1) % 50 == 0:
            print(
                f"[{idx + 1}/{args.sample_num}] "
                f"success={stats['success']}, fail={stats['fail']}"
            )

    # --- Save ---
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(json_all, f, ensure_ascii=False, indent=2)

    print(f"\nDone! {len(json_all)} samples -> {args.output_json}")
    print(f"  success: {stats['success']}, fail: {stats['fail']}")
    print(f"  checkpoint images: {checkpoint_dir}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build grid CoT dataset with dense visual checkpoints"
    )
    # Search params
    parser.add_argument(
        "--search_algo",
        type=str,
        choices=["bfs", "dfs", "beam"],
        default="bfs",
        help="Teacher search algorithm used to generate final trace.",
    )
    parser.add_argument("--sample_num", type=int, default=500)
    parser.add_argument("--checkpoint_interval", type=int, default=2,
                        help="Render one checkpoint every N reasoning steps")
    parser.add_argument("--candidate_num", type=int, default=4)
    parser.add_argument("--beam_num", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument(
        "--keep_failed",
        action="store_true",
        help="Keep unresolved samples with '<answer>fail to find a valid path</answer>'.",
    )
    # Grid params
    parser.add_argument("--grid_rows", type=int, default=4)
    parser.add_argument("--grid_cols", type=int, default=5)
    parser.add_argument("--map_p", type=float, default=0.8)
    parser.add_argument("--map_seed", type=int, default=42)
    parser.add_argument("--start_idx", type=int, default=0)
    # Paths
    parser.add_argument("--image_dir", type=str,
                        default=str(BAGEL_OUTPUT_DIR / "base_images"))
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_json", type=str,
                        default=str(BAGEL_OUTPUT_DIR / "frozen_lake_visual_cot_commit_beam.json"))
    parser.add_argument("--checkpoint_dir", type=str,
                        default=str(BAGEL_OUTPUT_DIR / "grid_images_commit_beam"))

    build_dataset(parser.parse_args())
