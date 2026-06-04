#!/usr/bin/env python3
"""Build MiniGrid navigation dataset with FrozenLake-style cumulative-path CoT."""

import argparse
import json
import math
import random
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Direction / action constants
# ---------------------------------------------------------------------------

DIR_TO_TEXT = {0: "east", 1: "south", 2: "west", 3: "north"}
ACTION_TO_TEXT = {"left": "turn left", "right": "turn right", "forward": "go forward"}
FORWARD_DELTAS = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
TURN_LEFT  = {0: 3, 1: 0, 2: 1, 3: 2}
TURN_RIGHT = {0: 1, 1: 2, 2: 3, 3: 0}
# BFS expansion order: try forward first, then turns
SEARCH_ACTION_ORDER = ("forward", "right", "left")


# ---------------------------------------------------------------------------
# Custom PIL renderer  — high-contrast style
#   background : dark grey  (64, 64, 64)
#   grid lines : darker grey (40, 40, 40)
#   wall       : orange + dark wavy lines  (230,120,30)
#   goal       : bright green  (50, 220, 50)
#   agent      : red triangle (arrow shows direction)
# ---------------------------------------------------------------------------

_BG      = (64,  64,  64)
_LINE    = (30,  30,  30)
_WALL_BG = (230, 120,  30)
_WALL_FG = (80,  40,   0)    # wave line colour
_GOAL    = (50,  220,  50)
_AGENT   = (220,  40,  40)

# agent direction → angle offset in degrees (PIL: 0=right, ccw positive)
# minigrid dirs: 0=east 1=south 2=west 3=north
_DIR_ANGLE = {0: 270, 1: 0, 2: 90, 3: 180}  # 0=north(up), 1=east(right), 2=south(down), 3=west(left)


def _draw_wavy_lines(draw, x0, y0, x1, y1, color, n_waves=3, thickness=2):
    """Draw horizontal wavy lines inside the rect [x0,y0,x1,y1]."""
    h = y1 - y0
    w = x1 - x0
    for k in range(1, n_waves + 1):
        cy = y0 + h * k // (n_waves + 1)
        pts = []
        steps = max(w // 3, 4)
        for i in range(steps + 1):
            t  = i / steps
            px = x0 + int(t * w)
            py = cy + int(math.sin(t * 2 * math.pi) * h * 0.07)
            pts.append((px, py))
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=color, width=thickness)


def _draw_agent_arrow(draw, cx, cy, cell_px, direction):
    """Draw a filled red triangle (arrow) centred at (cx,cy), pointing in direction."""
    r  = cell_px * 0.35
    angle_base = _DIR_ANGLE[direction]   # degrees; 0 = pointing right

    def pt(deg):
        rad = math.radians(angle_base + deg)
        return (cx + r * math.cos(rad), cy + r * math.sin(rad))

    tip   = pt(0)
    left  = pt(150)
    right = pt(210)
    draw.polygon([tip, left, right], fill=_AGENT)


def render_grid(rows, cols, walls, goal_pos, agent_pos, agent_dir, image_size):
    """Render the grid using PIL with high-contrast custom style."""
    margin   = max(4, image_size // (rows + 2) // 2)
    cell_px  = (image_size - 2 * margin) // max(rows, cols)
    actual_w = cell_px * cols + 2 * margin
    actual_h = cell_px * rows + 2 * margin

    img  = Image.new("RGB", (actual_w, actual_h), _BG)
    draw = ImageDraw.Draw(img)

    wall_set = {tuple(w) for w in walls}

    for r in range(rows):
        for c in range(cols):
            x0 = margin + c * cell_px
            y0 = margin + r * cell_px
            x1 = x0 + cell_px
            y1 = y0 + cell_px

            pos = (r, c)
            if pos == tuple(goal_pos):
                draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=_GOAL)
            elif pos in wall_set:
                draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=_WALL_BG)
                _draw_wavy_lines(draw, x0 + 2, y0 + 2, x1 - 2, y1 - 2,
                                 color=_WALL_FG, n_waves=3, thickness=max(1, cell_px // 20))
            # grid lines
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=_LINE, width=max(1, cell_px // 20))

    # Draw agent
    ar, ac = agent_pos
    cx = margin + ac * cell_px + cell_px // 2
    cy = margin + ar * cell_px + cell_px // 2
    _draw_agent_arrow(draw, cx, cy, cell_px, agent_dir)

    return img.resize((image_size, image_size), resample=Image.NEAREST)



# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def rotate_left(d):  return TURN_LEFT[d]
def rotate_right(d): return TURN_RIGHT[d]

def step_forward(pos, direction, rows, cols, walls):
    dr, dc = FORWARD_DELTAS[direction]
    nr, nc = pos[0] + dr, pos[1] + dc
    if not (0 <= nr < rows and 0 <= nc < cols):
        return pos, True
    if (nr, nc) in walls:
        return pos, True
    return (nr, nc), False

def apply_action(pos, direction, action, rows, cols, walls):
    if action == "left":
        return pos, rotate_left(direction), False
    if action == "right":
        return pos, rotate_right(direction), False
    next_pos, blocked = step_forward(pos, direction, rows, cols, walls)
    return next_pos, direction, blocked

def l1(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------------------------------------------------------------------
# BFS solver  (same as v1)
# ---------------------------------------------------------------------------

def bfs_solve(start_pos, start_dir, goal_pos, rows, cols, walls):
    start = (start_pos[0], start_pos[1], start_dir)
    queue = deque([(start, [])])
    visited = {start}
    while queue:
        (r, c, d), trace = queue.popleft()
        if (r, c) == goal_pos:
            return trace
        for action in SEARCH_ACTION_ORDER:
            if action == "left":
                nxt = (r, c, rotate_left(d))
            elif action == "right":
                nxt = (r, c, rotate_right(d))
            else:
                npos, blocked = step_forward((r, c), d, rows, cols, walls)
                if blocked:
                    continue
                nxt = (npos[0], npos[1], d)
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, trace + [action]))
    return None


# ---------------------------------------------------------------------------
# CoT text builders  (FrozenLake-style: cumulative path prefix everywhere)
# ---------------------------------------------------------------------------

def action_seq_text(actions):
    """Convert a list of action keys to a human-readable cumulative trace string."""
    if not actions:
        return "(start)"
    return " -> ".join(ACTION_TO_TEXT[a] for a in actions)


def wall_text(walls):
    if not walls:
        return "luckily there are no wall tiles on this map"
    coords = ", ".join(f"({r},{c})" for r, c in sorted(walls))
    if len(walls) == 1:
        return f"there is 1 wall tile I must avoid: {coords}"
    return f"there are {len(walls)} wall tiles I must avoid: {coords}"


def pre_definition(start_pos, start_dir, goal_pos, walls, rows, cols, sample_idx):
    """Opening paragraph — mirrors FrozenLake pre_definition style.

    Every template must contain:
      - "Starting position is (r, c), facing <dir>"  ← matched by START_PATTERN + START_DIR_PATTERN
      - "goal is at (r, c)"                           ← matched by GOAL_PATTERN
      - "wall tiles I must avoid: ..."                ← matched by WALLS_PATTERN  (or no-wall phrase)
    so that data.py/_parse_minigrid_state() can reliably extract all fields.
    """
    wall_intro = wall_text(walls)
    start_str  = f"({start_pos[0]}, {start_pos[1]})"
    goal_str   = f"({goal_pos[0]}, {goal_pos[1]})"
    dir_str    = DIR_TO_TEXT[start_dir]

    # Canonical anchor line always present — parser hooks onto these phrases.
    anchor = (
        f"Starting position is {start_str}, facing {dir_str}. "
        f"The goal is at {goal_str}. "
        f"{wall_intro.capitalize()}."
    )

    templates = [
        (
            f"Okay, I need to navigate an agent on this {rows}x{cols} grid.\n"
            f"{anchor}\n"
            f"Actions available: turn left, turn right, go forward.\n"
            f"My plan: at each step I'll try all three actions, simulate where each leads, "
            f"and keep the option that gets me closest to the goal.\n"
            f"Let me work through this."
        ),
        (
            f"Alright — I'm navigating a {rows}x{cols} MiniGrid map.\n"
            f"{anchor}\n"
            f"I'll reason step by step, testing each candidate action and committing to the best one."
        ),
        (
            f"This is a MiniGrid navigation task on a {rows}x{cols} grid.\n"
            f"{anchor}\n"
            f"I'll explore candidate actions at each step, simulate outcomes, and follow the path "
            f"that minimises distance to the goal while avoiding walls."
        ),
        (
            f"Let me find a path on this {rows}x{cols} grid.\n"
            f"{anchor}\n"
            f"I'll test turn left, turn right, and go forward at each decision point and pick the best."
        ),
    ]
    return templates[sample_idx % len(templates)]


# Candidate description styles — mirrors FrozenLake step_cot styles
_CANDIDATE_STYLES = [
    dict(
        prefix  = lambda trace, pos, d: f"If I take {trace}, I land at {pos} facing {d}",
        safe    = lambda n: f"still {n} step(s) from the goal",
        goal    = lambda:   "and that reaches the goal",
        blocked = lambda:   "but the way is blocked — not an option",
    ),
    dict(
        prefix  = lambda trace, pos, d: f"Taking {trace} puts me at {pos} facing {d}",
        safe    = lambda n: f"leaving {n} step(s) to go",
        goal    = lambda:   "which is exactly the goal",
        blocked = lambda:   "— but it's blocked, so this is off the table",
    ),
    dict(
        prefix  = lambda trace, pos, d: f"{trace} leads to {pos} facing {d}",
        safe    = lambda n: f"{n} step(s) remaining",
        goal    = lambda:   "the goal — perfect",
        blocked = lambda:   "a wall — I'll skip this",
    ),
    dict(
        prefix  = lambda trace, pos, d: f"Following {trace} I'd be at {pos} facing {d}",
        safe    = lambda n: f"with {n} step(s) still ahead",
        goal    = lambda:   "which means I've reached the goal",
        blocked = lambda:   "— that's a wall, ruled out",
    ),
    dict(
        prefix  = lambda trace, pos, d: f"Going {trace} brings me to {pos} facing {d}",
        safe    = lambda n: f"{n} move(s) short of the goal",
        goal    = lambda:   "and the goal is right there",
        blocked = lambda:   "— blocked by a wall, so no",
    ),
]


def step_cot(step_idx, committed_prefix, pos, direction,
             goal_pos, rows, cols, walls, chosen_action, sample_idx):
    """
    Generate one-step CoT text.

    Uses cumulative path prefix (like FrozenLake): every candidate trace string
    starts with the already-committed actions, so the model always sees the full
    path so far and can copy it straight into <answer>.
    """
    style = _CANDIDATE_STYLES[(step_idx + sample_idx) % len(_CANDIDATE_STYLES)]

    # --- evaluate all three actions ---
    snippets = []
    chosen_result = None

    actions = ["left", "right", "forward"]
    for action in actions:
        candidate_prefix = committed_prefix + [action]
        npos, ndir, blocked = apply_action(pos, direction, action, rows, cols, walls)
        trace_text = action_seq_text(candidate_prefix)
        prefix_str = style["prefix"](trace_text, f"({npos[0]},{npos[1]})", DIR_TO_TEXT[ndir])

        if blocked:
            status = style["blocked"]()
        elif npos == goal_pos:
            status = style["goal"]()
        else:
            dist = l1(npos, goal_pos)
            status = style["safe"](dist)

        snippets.append(f"{prefix_str} — {status}.")

        if action == chosen_action:
            chosen_result = (npos, ndir, blocked)

    # --- conclusion ---
    chosen_trace_text = action_seq_text(committed_prefix + [chosen_action])
    npos, ndir, blocked = chosen_result

    if npos == goal_pos:
        conclusion_variants = [
            f"I'll go with {chosen_trace_text} — it reaches the goal.",
            f"{chosen_trace_text} gets me to the goal, so that's my pick.",
            f"Clear winner: {chosen_trace_text} hits the goal.",
        ]
    else:
        n_left = l1(npos, goal_pos)
        conclusion_variants = [
            f"{chosen_trace_text} looks best — only {n_left} step(s) from the goal.",
            f"I'll continue with {chosen_trace_text}; it's the closest, with {n_left} step(s) to go.",
            f"{chosen_trace_text} gets me the furthest, leaving just {n_left} step(s). I'll keep this.",
            f"Going with {chosen_trace_text} for now — {n_left} more step(s) to the goal.",
        ]

    conclusion = conclusion_variants[(step_idx + sample_idx) % len(conclusion_variants)]
    snippets.append(conclusion)
    return " ".join(snippets)


def visual_check_text(pos, direction, goal_pos, walls, checkpoint_idx, sample_idx):
    """Visual-check line inserted after each <grid_token>."""
    r, c  = pos
    dist  = l1(pos, goal_pos)
    nearby_walls = [
        (r + dr, c + dc)
        for dr, dc in FORWARD_DELTAS.values()
        if (r + dr, c + dc) in walls
    ]

    if nearby_walls:
        wall_str   = str(sorted(nearby_walls))
        warnings   = [
            f"Watch out — wall(s) adjacent at {wall_str}.",
            f"Careful — nearby wall(s) at {wall_str}.",
            f"Note: wall(s) next to me at {wall_str}.",
        ]
        warning = warnings[(checkpoint_idx + sample_idx) % len(warnings)]
    else:
        clears  = [
            "Path ahead looks clear.",
            "No walls nearby — safe to continue.",
            "Surroundings are clear.",
        ]
        warning = clears[(checkpoint_idx + sample_idx) % len(clears)]

    templates = [
        f"(Visual check: I'm at ({r},{c}) facing {DIR_TO_TEXT[direction]}, {dist} step(s) from goal. {warning} Continuing.)",
        f"(Checking the grid: position ({r},{c}) facing {DIR_TO_TEXT[direction]}, distance to goal = {dist}. {warning} Moving on.)",
        f"(Grid snapshot: currently at ({r},{c}) facing {DIR_TO_TEXT[direction]}, {dist} step(s) left. {warning} Proceeding.)",
    ]
    return templates[(checkpoint_idx + sample_idx) % len(templates)]


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

def make_one_sample(rng, rows, cols, min_walls, max_walls):
    free_cells = [(r, c) for r in range(rows) for c in range(cols)]
    start      = rng.choice(free_cells)
    goal       = rng.choice([x for x in free_cells if x != start])
    start_dir  = rng.randrange(4)

    wall_candidates = [x for x in free_cells if x not in {start, goal}]
    rng.shuffle(wall_candidates)
    wall_count = min(rng.randint(min_walls, max_walls), len(wall_candidates))
    walls      = set(wall_candidates[:wall_count])

    trace = bfs_solve(start, start_dir, goal, rows, cols, walls)
    if trace is None:
        return None

    return {"start": start, "goal": goal, "start_dir": start_dir,
            "walls": walls, "trace": trace}


def build_sample_cot(sample, rows, cols, sample_idx, checkpoint_interval, image_size,
                     base_image_path, ckpt_dir):
    """
    Build the full CoT string and checkpoint image list for one sample.
    Returns (cot_text, checkpoint_image_paths).
    """
    start     = sample["start"]
    goal      = sample["goal"]
    start_dir = sample["start_dir"]
    walls     = sample["walls"]
    trace     = sample["trace"]

    lines = [pre_definition(start, start_dir, goal, walls, rows, cols, sample_idx)]

    pos       = start
    direction = start_dir
    checkpoint_images = []
    checkpoint_idx    = 0

    for step_idx, chosen_action in enumerate(trace):
        # one-step CoT with cumulative prefix
        lines.append(step_cot(
            step_idx        = step_idx,
            committed_prefix= list(trace[:step_idx]),
            pos             = pos,
            direction       = direction,
            goal_pos        = goal,
            rows            = rows,
            cols            = cols,
            walls           = walls,
            chosen_action   = chosen_action,
            sample_idx      = sample_idx,
        ))

        pos, direction, _ = apply_action(pos, direction, chosen_action, rows, cols, walls)

        # checkpoint after every N steps, and always after the final step
        # NOTE: we do NOT save separate checkpoint images — all <grid_token>s
        # reuse the base image (same as FrozenLake), so training and inference
        # see the same number of images.
        if (step_idx + 1) % checkpoint_interval == 0 or (step_idx + 1) == len(trace):
            lines.append("<grid_token>")
            lines.append(visual_check_text(pos, direction, goal, walls, checkpoint_idx, sample_idx))
            checkpoint_idx += 1

    final_answer = action_seq_text(trace)
    cot = "\n".join(["<think>", *lines, "</think>",
                     f"<answer>Final chosen trace: {final_answer}</answer>"])
    return cot


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(args):
    rng      = random.Random(args.seed)
    rows     = cols = args.grid_size
    image_dir = Path(args.image_dir)
    ckpt_dir  = Path(args.checkpoint_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    output    = []
    built     = 0
    sample_id = args.start_idx

    while built < args.sample_num:
        sample = make_one_sample(rng, rows, cols, args.min_walls, args.max_walls)
        if sample is None:
            continue

        base_image_path = image_dir / f"{sample_id}.jpg"
        render_grid(
            rows, cols, sample["walls"], sample["goal"],
            sample["start"], sample["start_dir"], args.image_size,
        ).save(base_image_path, format="JPEG", quality=95, subsampling=0)

        cot = build_sample_cot(
            sample             = sample,
            rows               = rows,
            cols               = cols,
            sample_idx         = built,
            checkpoint_interval= args.checkpoint_interval,
            image_size         = args.image_size,
            base_image_path    = base_image_path,
            ckpt_dir           = ckpt_dir,
        )

        output.append({
            "conversations": [
                {"from": "human", "value": "<image>"},
                {"from": "gpt",   "value": cot},
            ],
            "images": [str(base_image_path)],  # only base image, like FrozenLake
            "meta": {
                "env_type":  "minigrid_nav",
                "rows":      rows,
                "cols":      cols,
                "start":     list(sample["start"]),
                "start_dir": DIR_TO_TEXT[sample["start_dir"]],
                "goal":      list(sample["goal"]),
                "walls":     [list(w) for w in sorted(sample["walls"])],
            },
        })

        built     += 1
        sample_id += 1
        if built % 50 == 0:
            print(f"[{built}/{args.sample_num}] built")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"saved {out_path}  ({len(output)} samples)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-num",           type=int, default=500)
    p.add_argument("--grid-size",            type=int, default=4)
    p.add_argument("--min-walls",            type=int, default=1)
    p.add_argument("--max-walls",            type=int, default=3)
    p.add_argument("--checkpoint-interval",  type=int, default=2)
    p.add_argument("--seed",                 type=int, default=1234)
    p.add_argument("--image-size",           type=int, default=256)
    p.add_argument("--start-idx",            type=int, default=0)
    p.add_argument("--output-json",          required=True)
    p.add_argument("--image-dir",            required=True)
    p.add_argument("--checkpoint-dir",       required=True)
    return p.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
