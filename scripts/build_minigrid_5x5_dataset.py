#!/usr/bin/env python3
"""Build a MiniGrid DoorKey-style global-view dataset compatible with GridCoT."""

import argparse
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image
from minigrid.core.actions import Actions
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv


DIR_TO_TEXT = {
    0: "north",
    1: "east",
    2: "south",
    3: "west",
}

ACTION_TO_TEXT = {
    "left": "turn left",
    "right": "turn right",
    "forward": "go forward",
    "pickup": "pickup key",
    "toggle": "toggle door",
}

FORWARD_DELTAS = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}

TURN_LEFT = {0: 3, 1: 0, 2: 1, 3: 2}
TURN_RIGHT = {0: 1, 1: 2, 2: 3, 3: 0}
SEARCH_ACTION_ORDER = ("forward", "pickup", "toggle", "right", "left")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-num", type=int, default=1000)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--min-walls", type=int, default=2)
    parser.add_argument("--max-walls", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--keep-failed", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    return parser.parse_args()


class FixedMiniGrid5x5Env(MiniGridEnv):
    def __init__(
        self,
        logical_size,
        start_pos,
        start_dir,
        goal_pos,
        walls,
        key_pos,
        door_pos,
        door_locked=True,
        door_open=False,
        carrying_key=False,
        tile_size=32,
        render_mode="rgb_array",
    ):
        mission_space = MissionSpace(mission_func=lambda: "pick up the key, open the door, and reach the goal")
        self.logical_size = logical_size
        self.fixed_start_pos = tuple(start_pos)
        self.fixed_start_dir = int(start_dir)
        self.fixed_goal_pos = tuple(goal_pos)
        self.fixed_walls = {tuple(w) for w in walls}
        self.fixed_key_pos = tuple(key_pos) if key_pos is not None else None
        self.fixed_door_pos = tuple(door_pos) if door_pos is not None else None
        self.fixed_door_locked = bool(door_locked)
        self.fixed_door_open = bool(door_open)
        self.fixed_carrying_key = bool(carrying_key)
        super().__init__(
            mission_space=mission_space,
            width=logical_size + 2,
            height=logical_size + 2,
            max_steps=4 * logical_size * logical_size,
            see_through_walls=True,
            agent_view_size=7,
            render_mode=render_mode,
            highlight=False,
            tile_size=tile_size,
            agent_pov=False,
        )

    def _to_env(self, pos):
        return pos[1] + 1, pos[0] + 1

    def _to_logical(self, env_pos):
        return env_pos[1] - 1, env_pos[0] - 1

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        for wall in self.fixed_walls:
            c, r = self._to_env(wall)
            self.put_obj(Wall(), c, r)

        if self.fixed_door_pos is not None:
            door_c, door_r = self._to_env(self.fixed_door_pos)
            self.put_obj(
                Door(color="yellow", is_open=self.fixed_door_open, is_locked=self.fixed_door_locked),
                door_c,
                door_r,
            )

        if self.fixed_key_pos is not None:
            key_c, key_r = self._to_env(self.fixed_key_pos)
            self.put_obj(Key(color="yellow"), key_c, key_r)

        goal_c, goal_r = self._to_env(self.fixed_goal_pos)
        self.put_obj(Goal(), goal_c, goal_r)

        start_c, start_r = self._to_env(self.fixed_start_pos)
        self.agent_pos = np.array((start_c, start_r))
        # Convert our direction convention to MiniGrid's:
        # Ours: 0=north, 1=east, 2=south, 3=west
        # MiniGrid: 0=right(east), 1=down(south), 2=left(west), 3=up(north)
        _LOGICAL_TO_MINIGRID_DIR = {0: 3, 1: 0, 2: 1, 3: 2}
        self.agent_dir = _LOGICAL_TO_MINIGRID_DIR[self.fixed_start_dir]
        if self.fixed_carrying_key:
            self.carrying = Key(color="yellow")
            self.carrying.cur_pos = np.array([-1, -1])
        self.mission = "pick up the key, open the door, and reach the goal"


def rotate_left(direction):
    return TURN_LEFT[direction]


def rotate_right(direction):
    return TURN_RIGHT[direction]


def step_forward(pos, direction, rows, cols, walls, door_pos, door_open):
    dr, dc = FORWARD_DELTAS[direction]
    nr, nc = pos[0] + dr, pos[1] + dc
    if not (0 <= nr < rows and 0 <= nc < cols):
        return pos, True
    if (nr, nc) in walls:
        return pos, True
    if (nr, nc) == door_pos and not door_open:
        return pos, True
    return (nr, nc), False


def shortest_pos_distance(start, goal, rows, cols, walls):
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        pos, dist = queue.popleft()
        if pos == goal:
            return dist
        for dr, dc in FORWARD_DELTAS.values():
            nxt = (pos[0] + dr, pos[1] + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
                continue
            if nxt in walls or nxt in visited:
                continue
            visited.add(nxt)
            queue.append((nxt, dist + 1))
    return None


def front_cell(pos, direction, rows, cols):
    dr, dc = FORWARD_DELTAS[direction]
    nr, nc = pos[0] + dr, pos[1] + dc
    if not (0 <= nr < rows and 0 <= nc < cols):
        return None
    return (nr, nc)


def plan_text(agent_pos, key_pos, door_pos, goal_pos, carrying_key, door_open):
    if not carrying_key:
        return f"Current plan: move toward the key at ({key_pos[0]}, {key_pos[1]})."
    if not door_open:
        return f"Current plan: move toward the door at ({door_pos[0]}, {door_pos[1]}) and open it."
    return f"Current plan: move toward the goal at ({goal_pos[0]}, {goal_pos[1]})."


def stage_name(carrying_key, door_open):
    if not carrying_key:
        return "key"
    if not door_open:
        return "door"
    return "goal"


def candidate_actions(pos, direction, rows, cols, key_pos, door_pos, carrying_key, key_present, door_open):
    actions = ["left", "right", "forward"]
    front = front_cell(pos, direction, rows, cols)
    if key_present and not carrying_key and front == key_pos:
        actions.append("pickup")
    if carrying_key and not door_open and front == door_pos:
        actions.append("toggle")
    return actions


def bfs_turn_forward(start_pos, start_dir, goal_pos, rows, cols, walls, key_pos, door_pos):
    start = (start_pos[0], start_pos[1], start_dir, False, True, False)
    queue = deque([(start, [])])
    visited = {start}
    while queue:
        (r, c, direction, carrying_key, key_present, door_open), trace = queue.popleft()
        if (r, c) == goal_pos:
            return trace
        for action in SEARCH_ACTION_ORDER:
            if action == "left":
                nxt = (r, c, rotate_left(direction), carrying_key, key_present, door_open)
            elif action == "right":
                nxt = (r, c, rotate_right(direction), carrying_key, key_present, door_open)
            elif action == "pickup":
                target = front_cell((r, c), direction, rows, cols)
                if not key_present or carrying_key or target != key_pos:
                    continue
                nxt = (r, c, direction, True, False, door_open)
            elif action == "toggle":
                target = front_cell((r, c), direction, rows, cols)
                if target != door_pos:
                    continue
                if door_open:
                    continue
                if not carrying_key:
                    continue
                nxt = (r, c, direction, carrying_key, key_present, True)
            else:
                next_pos, blocked = step_forward((r, c), direction, rows, cols, walls, door_pos, door_open)
                if blocked:
                    continue
                nxt = (next_pos[0], next_pos[1], direction, carrying_key, key_present, door_open)
            if nxt in visited:
                continue
            visited.add(nxt)
            queue.append((nxt, trace + [action]))
    return None


def apply_action(pos, direction, action, rows, cols, walls, key_pos, door_pos, carrying_key, key_present, door_open):
    if action == "left":
        return pos, rotate_left(direction), carrying_key, key_present, door_open, False
    if action == "right":
        return pos, rotate_right(direction), carrying_key, key_present, door_open, False
    if action == "pickup":
        target = front_cell(pos, direction, rows, cols)
        blocked = not key_present or carrying_key or target != key_pos
        if blocked:
            return pos, direction, carrying_key, key_present, door_open, True
        return pos, direction, True, False, door_open, False
    if action == "toggle":
        target = front_cell(pos, direction, rows, cols)
        blocked = target != door_pos or door_open or (not carrying_key)
        if blocked:
            return pos, direction, carrying_key, key_present, door_open, True
        return pos, direction, carrying_key, key_present, True, False
    next_pos, blocked = step_forward(pos, direction, rows, cols, walls, door_pos, door_open)
    return next_pos, direction, carrying_key, key_present, door_open, blocked


def render_grid(rows, cols, walls, goal_pos, key_pos, door_pos, door_locked, door_open, agent_pos, agent_dir, carrying_key, image_size):
    env = FixedMiniGrid5x5Env(
        logical_size=rows,
        start_pos=agent_pos,
        start_dir=agent_dir,
        goal_pos=goal_pos,
        walls=walls,
        key_pos=None if carrying_key else key_pos,
        door_pos=door_pos,
        door_locked=door_locked,
        door_open=door_open,
        carrying_key=carrying_key,
        tile_size=max(8, image_size // (rows + 2)),
        render_mode="rgb_array",
    )
    env.reset(seed=0)
    frame = env.get_frame(agent_pov=False, highlight=False, tile_size=max(8, image_size // (rows + 2)))
    env.close()
    return Image.fromarray(frame).convert("RGB").resize((image_size, image_size), resample=Image.NEAREST)


def wall_text(walls):
    if not walls:
        return "There are no wall tiles blocking the route."
    coords = ", ".join(f"({r},{c})" for r, c in sorted(walls))
    if len(walls) == 1:
        return f"There is 1 wall tile I must avoid: {coords}."
    return f"There are {len(walls)} wall tiles I must avoid: {coords}."


def build_intro(rows, cols, start_pos, start_dir, goal_pos, walls, key_pos, door_pos):
    return [
        f"This is a MiniGrid-style DoorKey navigation problem on a {rows}x{cols} grid.",
        f"Starting position is ({start_pos[0]}, {start_pos[1]}), facing {DIR_TO_TEXT[start_dir]}.",
        f"The goal is at ({goal_pos[0]}, {goal_pos[1]}).",
        f"The key is at ({key_pos[0]}, {key_pos[1]}).",
        f"The locked yellow door is at ({door_pos[0]}, {door_pos[1]}).",
        wall_text(walls),
        "The action space is: turn left, turn right, go forward, pickup key, toggle door.",
        "I need to pick up the key, unlock/open the door, avoid wall tiles, and produce a valid action sequence to the goal.",
    ]


def chosen_step_text(action, next_pos, next_dir, goal_pos, carrying_key, door_open):
    action_text = ACTION_TO_TEXT[action]
    dist = abs(goal_pos[0] - next_pos[0]) + abs(goal_pos[1] - next_pos[1])
    if action == "pickup":
        return (
            f"If I pickup key now, I remain at "
            f"({next_pos[0]}, {next_pos[1]}) facing {DIR_TO_TEXT[next_dir]}. "
            f"The position-distance to the goal becomes {dist}, so I pick it up."
        )
    if action == "toggle":
        return (
            f"If I toggle door now, I remain at "
            f"({next_pos[0]}, {next_pos[1]}) facing {DIR_TO_TEXT[next_dir]}. "
            f"The position-distance to the goal becomes {dist}, so I open it."
        )
    return (
        f"If I take {action_text}, I will be at ({next_pos[0]}, {next_pos[1]}) facing "
        f"{DIR_TO_TEXT[next_dir]}. The position-distance to the goal becomes {dist}, "
        f"so I take {action_text}."
    )


def binary_choice_text(chosen_action, alt_action, chosen_state, alt_state, goal_pos):
    chosen_pos, chosen_dir, _, _, _, _ = chosen_state
    alt_pos, alt_dir, _, _, _, _ = alt_state
    chosen_dist = abs(goal_pos[0] - chosen_pos[0]) + abs(goal_pos[1] - chosen_pos[1])
    alt_dist = abs(goal_pos[0] - alt_pos[0]) + abs(goal_pos[1] - alt_pos[1])
    chosen_text = ACTION_TO_TEXT[chosen_action]
    alt_text = ACTION_TO_TEXT[alt_action]

    if chosen_action == "pickup":
        chosen_clause = (
            f"If I pickup key now, I remain at ({chosen_pos[0]}, {chosen_pos[1]}) facing "
            f"{DIR_TO_TEXT[chosen_dir]}"
        )
    elif chosen_action == "toggle":
        chosen_clause = (
            f"If I toggle door now, I remain at ({chosen_pos[0]}, {chosen_pos[1]}) facing "
            f"{DIR_TO_TEXT[chosen_dir]}"
        )
    else:
        chosen_clause = (
            f"If I take {chosen_text}, I will be at ({chosen_pos[0]}, {chosen_pos[1]}) facing "
            f"{DIR_TO_TEXT[chosen_dir]}"
        )

    if alt_action == "pickup":
        alt_clause = (
            f"if I pickup key instead, I remain at ({alt_pos[0]}, {alt_pos[1]}) facing "
            f"{DIR_TO_TEXT[alt_dir]}"
        )
    elif alt_action == "toggle":
        alt_clause = (
            f"if I toggle door instead, I remain at ({alt_pos[0]}, {alt_pos[1]}) facing "
            f"{DIR_TO_TEXT[alt_dir]}"
        )
    else:
        alt_clause = (
            f"if I take {alt_text} instead, I will be at ({alt_pos[0]}, {alt_pos[1]}) facing "
            f"{DIR_TO_TEXT[alt_dir]}"
        )

    if chosen_dist < alt_dist:
        reason = "it gets me closer to the goal"
    elif chosen_dist > alt_dist:
        reason = "it avoids moving away from the goal"
    else:
        reason = "it is the right move here"
    if chosen_action == "pickup":
        decision = "so I pickup key."
    elif chosen_action == "toggle":
        decision = "so I toggle door."
    else:
        decision = f"so I take {chosen_text}."
    return f"{chosen_clause}; {alt_clause}. {chosen_text} is better here because {reason}, {decision}"


def pick_alternative_action(actions, chosen_action):
    for action in actions:
        if action != chosen_action:
            return action
    if chosen_action == "left":
        return "right"
    if chosen_action == "right":
        return "left"
    if chosen_action == "forward":
        return "left"
    if chosen_action == "pickup":
        return "forward"
    return "forward"


def candidate_text(prefix_actions, action, next_pos, next_dir, goal_pos, blocked, carrying_key, key_present, door_open):
    action_text = ACTION_TO_TEXT[action]
    if blocked:
        return (
            f"If I add {action_text}, I stay at ({next_pos[0]}, {next_pos[1]}) facing "
            f"{DIR_TO_TEXT[next_dir]} because that action is not currently valid."
        )
    dist = abs(goal_pos[0] - next_pos[0]) + abs(goal_pos[1] - next_pos[1])
    if action == "pickup":
        return (
            f"I am next to the key now, so I pick it up and remain at "
            f"({next_pos[0]}, {next_pos[1]}) facing {DIR_TO_TEXT[next_dir]} with "
            f"{dist} step(s) of position-distance left to the goal."
        )
    if action == "toggle":
        return (
            f"I am in front of the door now, so I open it and remain at "
            f"({next_pos[0]}, {next_pos[1]}) facing {DIR_TO_TEXT[next_dir]} with "
            f"{dist} step(s) of position-distance left to the goal."
        )
    return (
        f"If I add {action_text}, I end up at ({next_pos[0]}, {next_pos[1]}) facing "
        f"{DIR_TO_TEXT[next_dir]} with {dist} step(s) of position-distance left to the goal."
    )


def prefix_text(prefix_actions):
    if not prefix_actions:
        return "Current committed prefix: start."
    return f"Current committed prefix: {' -> '.join(ACTION_TO_TEXT[a] for a in prefix_actions)}."


def visual_check_text(agent_pos, agent_dir, goal_pos, walls, key_pos, door_pos, carrying_key, key_present, door_open):
    r, c = agent_pos
    dist = abs(goal_pos[0] - r) + abs(goal_pos[1] - c)
    nearby = []
    for dr, dc in FORWARD_DELTAS.values():
        nr, nc = r + dr, c + dc
        if (nr, nc) in walls:
            nearby.append((nr, nc))
    if nearby:
        warning = f"Nearby wall tiles: {sorted(nearby)}."
    else:
        warning = "No adjacent wall tiles."
    key_text = "key already picked up" if carrying_key else f"key at ({key_pos[0]},{key_pos[1]})" if key_present else "key no longer on map"
    door_text = f"door at ({door_pos[0]},{door_pos[1]}) is {'open' if door_open else 'locked'}"
    return (
        f"(Visual check: I'm at ({r},{c}), facing {DIR_TO_TEXT[agent_dir]}, "
        f"{dist} step(s) from goal by position-distance. {warning} {key_text}. {door_text}. Continuing.)"
    )


def make_one_sample(rng, rows, cols, min_walls, max_walls):
    if rows < 4 or cols < 4:
        return None

    orientation = rng.choice(["vertical", "horizontal"])
    if orientation == "vertical":
        barrier = rng.randint(1, cols - 2)
        door_row = rng.randint(1, rows - 2)
        door_pos = (door_row, barrier)
        walls = {(r, barrier) for r in range(rows) if r != door_row}
        start_side = [(r, c) for r in range(rows) for c in range(barrier)]
        goal_side = [(r, c) for r in range(rows) for c in range(barrier + 1, cols)]
    else:
        barrier = rng.randint(1, rows - 2)
        door_col = rng.randint(1, cols - 2)
        door_pos = (barrier, door_col)
        walls = {(barrier, c) for c in range(cols) if c != door_col}
        start_side = [(r, c) for r in range(barrier) for c in range(cols)]
        goal_side = [(r, c) for r in range(barrier + 1, rows) for c in range(cols)]

    if not start_side or not goal_side:
        return None

    start = rng.choice(start_side)
    goal = rng.choice(goal_side)
    key_choices = [cell for cell in start_side if cell != start]
    if not key_choices:
        return None
    key_pos = rng.choice(key_choices)
    start_dir = rng.randrange(4)

    free_cells = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if (r, c) not in walls and (r, c) not in {start, goal, key_pos, door_pos}
    ]
    rng.shuffle(free_cells)
    extra_wall_budget = max(0, min(max_walls, len(free_cells)))
    extra_wall_count = min(rng.randint(min_walls, max_walls), extra_wall_budget)
    for candidate in free_cells[:extra_wall_count]:
        walls.add(candidate)

    trace = bfs_turn_forward(start, start_dir, goal, rows, cols, walls, key_pos, door_pos)
    if trace is None:
        return None

    return {
        "start": start,
        "goal": goal,
        "start_dir": start_dir,
        "walls": walls,
        "key_pos": key_pos,
        "door_pos": door_pos,
        "trace": trace,
    }


def build_dataset(args):
    rng = random.Random(args.seed)
    rows = cols = args.grid_size
    output = []
    image_dir = Path(args.image_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    built = 0
    sample_id = args.start_idx
    while built < args.sample_num:
        sample = make_one_sample(rng, rows, cols, args.min_walls, args.max_walls)
        if sample is None:
            continue

        base_image_path = image_dir / f"{sample_id}.jpg"
        render_grid(
            rows,
            cols,
            sample["walls"],
            sample["goal"],
            sample["key_pos"],
            sample["door_pos"],
            True,
            False,
            sample["start"],
            sample["start_dir"],
            False,
            args.image_size,
        ).save(base_image_path, format="JPEG", quality=95, subsampling=0)

        lines = build_intro(
            rows,
            cols,
            sample["start"],
            sample["start_dir"],
            sample["goal"],
            sample["walls"],
            sample["key_pos"],
            sample["door_pos"],
        )
        pos = sample["start"]
        direction = sample["start_dir"]
        carrying_key = False
        key_present = True
        door_open = False
        last_stage = None
        checkpoint_images = []

        for step_idx, chosen_action in enumerate(sample["trace"], start=1):
            current_stage = stage_name(carrying_key, door_open)
            if current_stage != last_stage:
                lines.append(prefix_text(sample["trace"][: step_idx - 1]))
                lines.append(
                    plan_text(
                        pos,
                        sample["key_pos"],
                        sample["door_pos"],
                        sample["goal"],
                        carrying_key,
                        door_open,
                    )
                )
                last_stage = current_stage

            available_actions = candidate_actions(
                pos,
                direction,
                rows,
                cols,
                sample["key_pos"],
                sample["door_pos"],
                carrying_key,
                key_present,
                door_open,
            )
            alt_action = pick_alternative_action(available_actions, chosen_action)
            next_pos, next_dir, next_carrying, next_key_present, next_door_open, blocked = apply_action(
                pos,
                direction,
                chosen_action,
                rows,
                cols,
                sample["walls"],
                sample["key_pos"],
                sample["door_pos"],
                carrying_key,
                key_present,
                door_open,
            )
            if blocked:
                break
            alt_state = apply_action(
                pos,
                direction,
                alt_action,
                rows,
                cols,
                sample["walls"],
                sample["key_pos"],
                sample["door_pos"],
                carrying_key,
                key_present,
                door_open,
            )
            lines.append(
                binary_choice_text(
                    chosen_action,
                    alt_action,
                    (next_pos, next_dir, next_carrying, next_key_present, next_door_open, blocked),
                    alt_state,
                    sample["goal"],
                )
            )
            pos, direction, carrying_key, key_present, door_open = (
                next_pos,
                next_dir,
                next_carrying,
                next_key_present,
                next_door_open,
            )

            if step_idx % args.checkpoint_interval == 0 or step_idx == len(sample["trace"]):
                ckpt_path = ckpt_dir / f"{sample_id}_step{step_idx}.png"
                render_grid(
                    rows,
                    cols,
                    sample["walls"],
                    sample["goal"],
                    sample["key_pos"],
                    sample["door_pos"],
                    not door_open,
                    door_open,
                    pos,
                    direction,
                    carrying_key,
                    args.image_size,
                ).save(ckpt_path)
                checkpoint_images.append(str(ckpt_path))
                lines.append("<grid_token>")
                lines.append(prefix_text(sample["trace"][:step_idx]))
                lines.append(
                    visual_check_text(
                        pos,
                        direction,
                        sample["goal"],
                        sample["walls"],
                        sample["key_pos"],
                        sample["door_pos"],
                        carrying_key,
                        key_present,
                        door_open,
                    )
                )

        final_answer = " -> ".join(ACTION_TO_TEXT[a] for a in sample["trace"])
        cot = "\n".join(["<think>", *lines, "</think>", f"<answer>Final chosen trace: {final_answer}</answer>"])

        output.append(
            {
                "conversations": [
                    {"from": "human", "value": "<image>"},
                    {"from": "gpt", "value": cot},
                ],
                "images": [str(base_image_path), *checkpoint_images],
                "meta": {
                    "env_type": "minigrid",
                    "rows": rows,
                    "cols": cols,
                    "start": list(sample["start"]),
                    "start_dir": DIR_TO_TEXT[sample["start_dir"]],
                    "goal": list(sample["goal"]),
                    "walls": [list(x) for x in sorted(sample["walls"])],
                    "key": list(sample["key_pos"]),
                    "door": list(sample["door_pos"]),
                },
            }
        )

        built += 1
        sample_id += 1
        if built % 50 == 0:
            print(f"[{built}/{args.sample_num}] built")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"saved {out_path}")
    print(f"samples {len(output)}")


def main():
    args = parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
