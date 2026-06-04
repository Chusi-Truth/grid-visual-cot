"""Data loading for grid CoT training. Handles grid token expansion and collation."""

import copy
import re
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
try:
    import ujson as json
except ImportError:
    import json

from constants import (
    IGNORE_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN, GRID_PAD_TOKEN, GRID_VISIBLE_TOKEN,
    GRID_SYSTEM_MESSAGE, GRID_SYSTEM_MESSAGE_TEXT_ONLY, GRID_SYSTEM_MESSAGE_STATE_DESC,
    MINIGRID_SYSTEM_MESSAGE, MINIGRID_SYSTEM_MESSAGE_TEXT_ONLY, MINIGRID_SYSTEM_MESSAGE_STATE_DESC,
    MINIGRID_NAV_SYSTEM_MESSAGE, MINIGRID_NAV_SYSTEM_MESSAGE_TEXT_ONLY, MINIGRID_NAV_SYSTEM_MESSAGE_STATE_DESC,
    GRID_TOKENS_PER_IMAGE, GRID_TOKEN_GROUP_PATTERN,
    infer_grid_size_from_text, build_grid_user_prompt,
)


# ---------------------------------------------------------------------------
# Grid token utilities
# ---------------------------------------------------------------------------

def make_grid_token_block(num_tokens=GRID_TOKENS_PER_IMAGE):
    """Create expanded token block: 1 visible + (num_tokens-1) pad."""
    if num_tokens <= 0:
        return ""
    if num_tokens == 1:
        return GRID_VISIBLE_TOKEN
    return GRID_VISIBLE_TOKEN + (GRID_PAD_TOKEN * (num_tokens - 1))


def count_grid_token_groups(text):
    return len(GRID_TOKEN_GROUP_PATTERN.findall(text))


def expand_grid_token_placeholders(text, num_tokens=GRID_TOKENS_PER_IMAGE):
    return GRID_TOKEN_GROUP_PATTERN.sub(make_grid_token_block(num_tokens), text)


def dropout_grid_token_groups(text, grid_image_paths, dropout_prob, rng):
    """Randomly drop grid token groups (and corresponding images) for regularization."""
    if dropout_prob <= 0 or len(grid_image_paths) == 0:
        return text, grid_image_paths

    matches = list(GRID_TOKEN_GROUP_PATTERN.finditer(text))
    if len(matches) != len(grid_image_paths):
        raise ValueError(
            f"Grid token group count {len(matches)} != grid image count {len(grid_image_paths)}."
        )
    if len(matches) <= 1:
        return text, grid_image_paths

    keep_flags = [rng.random() >= dropout_prob for _ in matches]
    if not any(keep_flags):
        keep_flags[int(rng.integers(len(matches)))] = True

    kept_parts, kept_images = [], []
    last_end = 0
    for keep, match, img_path in zip(keep_flags, matches, grid_image_paths):
        kept_parts.append(text[last_end:match.start()])
        if keep:
            kept_parts.append(match.group(0))
            kept_images.append(img_path)
        last_end = match.end()
    kept_parts.append(text[last_end:])
    return "".join(kept_parts), kept_images


def normalize_reasoning_tags(text):
    return text.replace("<\\think>", "</think>").replace("<\\answer>", "</answer>")


START_PATTERN = re.compile(
    r"(?:starting position is|I start at|starting at)\s*\((\d+),\s*(\d+)\)",
    re.IGNORECASE,
)
GOAL_PATTERN = re.compile(
    r"(?:goal(?: is| =)?(?: at)?|trying to reach|reach|get from\s*\(\d+,\s*\d+\)\s*to)\s*\((\d+),\s*(\d+)\)",
    re.IGNORECASE,
)
HOLES_PATTERN = re.compile(
    r"holes?(?: I must avoid)?\s*:\s*((?:\(\d+,\s*\d+\)\s*,?\s*)+)",
    re.IGNORECASE,
)
WALLS_PATTERN = re.compile(
    r"wall tiles?(?: I must avoid)?\s*:\s*((?:\(\d+,\s*\d+\)\s*,?\s*)+)",
    re.IGNORECASE,
)
KEY_PATTERN = re.compile(
    r"key(?: is| =)?(?: at)?\s*\((\d+),\s*(\d+)\)",
    re.IGNORECASE,
)
DOOR_PATTERN = re.compile(
    r"door(?: is| =)?(?: at)?\s*\((\d+),\s*(\d+)\)",
    re.IGNORECASE,
)
DOOR_STATE_PATTERN = re.compile(
    r"door at\s*\(\d+,\s*\d+\)\s*is\s*(open|locked)",
    re.IGNORECASE,
)
CARRYING_KEY_PATTERN = re.compile(
    r"carrying_key\s*=\s*(true|false)|key already picked up",
    re.IGNORECASE,
)
KEY_PRESENT_PATTERN = re.compile(
    r"key_present\s*=\s*(true|false)|key no longer on map",
    re.IGNORECASE,
)
CHECKPOINT_POS_PATTERN = re.compile(
    r"(?:I'm at|position|currently at)\s*\((\d+),\s*(\d+)\)",
    re.IGNORECASE,
)
START_DIR_PATTERN = re.compile(
    r"starting position is\s*\(\d+,\s*\d+\),\s*facing\s*(north|east|south|west)",
    re.IGNORECASE,
)
CHECKPOINT_DIR_PATTERN = re.compile(
    r"(?:I'm at|position|currently at)\s*\(\d+,\s*\d+\),\s*facing\s*(north|east|south|west)",
    re.IGNORECASE,
)
GRID_SIZE_PATTERN = re.compile(r"(\d+)x(\d+)\s+grid", re.IGNORECASE)
COORD_PATTERN = re.compile(r"\((\d+),\s*(\d+)\)")
DIR_TO_ID = {"north": 0, "east": 1, "south": 2, "west": 3}


def detect_env_type(text):
    lowered = (text or "").lower()
    if "minigrid-style navigation" in lowered:
        return "minigrid_nav"
    if "minigrid-style" in lowered or "turn left" in lowered or "go forward" in lowered:
        return "minigrid"
    return "frozenlake"


def _parse_start_goal_holes(text):
    start_match = START_PATTERN.search(text or "")
    start = (int(start_match.group(1)), int(start_match.group(2))) if start_match else (0, 0)

    goal_match = GOAL_PATTERN.search(text or "")
    if goal_match:
        goal = (int(goal_match.group(1)), int(goal_match.group(2)))
    else:
        all_coords = [(int(r), int(c)) for r, c in COORD_PATTERN.findall(text or "")]
        goal = max(all_coords) if all_coords else None

    holes_match = HOLES_PATTERN.search(text or "")
    holes = {tuple(map(int, xy)) for xy in COORD_PATTERN.findall(holes_match.group(1))} if holes_match else set()
    return start, goal, holes


def _parse_grid_size(text):
    match = GRID_SIZE_PATTERN.search(text or "")
    if match:
        return int(match.group(1)), int(match.group(2))
    coords = [(int(r), int(c)) for r, c in COORD_PATTERN.findall(text or "")]
    if not coords:
        return None, None
    return max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1


def _parse_minigrid_state(text):
    start_match = START_PATTERN.search(text or "")
    start = (int(start_match.group(1)), int(start_match.group(2))) if start_match else (0, 0)
    goal_match = GOAL_PATTERN.search(text or "")
    if goal_match:
        goal = (int(goal_match.group(1)), int(goal_match.group(2)))
    else:
        goal = None
    wall_match = WALLS_PATTERN.search(text or "")
    walls = {tuple(map(int, xy)) for xy in COORD_PATTERN.findall(wall_match.group(1))} if wall_match else set()
    key_match = KEY_PATTERN.search(text or "")
    key_pos = (int(key_match.group(1)), int(key_match.group(2))) if key_match else None
    door_match = DOOR_PATTERN.search(text or "")
    door_pos = (int(door_match.group(1)), int(door_match.group(2))) if door_match else None
    dir_match = START_DIR_PATTERN.search(text or "")
    start_dir = DIR_TO_ID[dir_match.group(1).lower()] if dir_match else 0
    rows, cols = _parse_grid_size(text)
    if rows is None or cols is None:
        row_vals = [start[0], goal[0] if goal else 0, *[w[0] for w in walls]]
        col_vals = [start[1], goal[1] if goal else 0, *[w[1] for w in walls]]
        if key_pos is not None:
            row_vals.append(key_pos[0])
            col_vals.append(key_pos[1])
        if door_pos is not None:
            row_vals.append(door_pos[0])
            col_vals.append(door_pos[1])
        rows = max(row_vals or [0]) + 1
        cols = max(col_vals or [0]) + 1
    return start, start_dir, goal, walls, key_pos, door_pos, rows, cols


def _extract_checkpoint_positions(text):
    positions = []
    for match in GRID_TOKEN_GROUP_PATTERN.finditer(text):
        suffix = text[match.end():]
        pos_match = CHECKPOINT_POS_PATTERN.search(suffix)
        if pos_match is None:
            raise ValueError("Found <grid_token> without a following checkpoint position description.")
        positions.append((int(pos_match.group(1)), int(pos_match.group(2))))
    return positions


def _extract_checkpoint_states(text):
    states = []
    for match in GRID_TOKEN_GROUP_PATTERN.finditer(text):
        suffix = text[match.end():]
        pos_match = CHECKPOINT_POS_PATTERN.search(suffix)
        dir_match = CHECKPOINT_DIR_PATTERN.search(suffix)
        if pos_match is None:
            raise ValueError("Found <grid_token> without a following checkpoint position description.")
        direction = DIR_TO_ID[dir_match.group(1).lower()] if dir_match else 0
        carrying_match = CARRYING_KEY_PATTERN.search(suffix)
        key_present_match = KEY_PRESENT_PATTERN.search(suffix)
        door_state_match = DOOR_STATE_PATTERN.search(suffix)
        carrying_key = False
        if carrying_match:
            if carrying_match.group(1) is not None:
                carrying_key = carrying_match.group(1).lower() == "true"
            else:
                carrying_key = True
        key_present = True
        if key_present_match:
            if key_present_match.group(1) is not None:
                key_present = key_present_match.group(1).lower() == "true"
            else:
                key_present = False
        door_open = bool(door_state_match and door_state_match.group(1).lower() == "open")
        states.append(((int(pos_match.group(1)), int(pos_match.group(2))), direction, carrying_key, key_present, door_open))
    return states


def _build_grid_state_tensor(agent_pos, goal_pos, holes, rows, cols):
    # Channels:
    # 0: agent
    # 1: goal
    # 2: hole mask
    # 3: safe-cell mask (non-hole)
    # 4: normalized row index
    # 5: normalized col index
    # 6: safe_up
    # 7: safe_down
    # 8: safe_left
    # 9: safe_right
    state = torch.zeros((10, rows, cols), dtype=torch.float32)
    hole_set = set(holes)

    for r in range(rows):
        for c in range(cols):
            state[4, r, c] = r / max(rows - 1, 1)
            state[5, r, c] = c / max(cols - 1, 1)
            if (r, c) in hole_set:
                state[2, r, c] = 1.0
            else:
                state[3, r, c] = 1.0
            # Local one-step safety priors for each direction.
            state[6, r, c] = float(r - 1 >= 0 and (r - 1, c) not in hole_set)      # up
            state[7, r, c] = float(r + 1 < rows and (r + 1, c) not in hole_set)    # down
            state[8, r, c] = float(c - 1 >= 0 and (r, c - 1) not in hole_set)      # left
            state[9, r, c] = float(c + 1 < cols and (r, c + 1) not in hole_set)    # right

    state[0, agent_pos[0], agent_pos[1]] = 1.0
    state[1, goal_pos[0], goal_pos[1]] = 1.0
    return state


def _build_minigrid_state_tensor(agent_pos, agent_dir, goal_pos, walls, key_pos, door_pos, carrying_key, key_present, door_open, rows, cols):
    # Channels:
    # 0: agent
    # 1: goal
    # 2: wall mask
    # 3: closed-door mask
    # 4: key mask
    # 5: carrying-key indicator at agent cell
    # 6-9: agent facing north/east/south/west (at agent cell only)
    state = torch.zeros((10, rows, cols), dtype=torch.float32)
    wall_set = set(walls)
    for r in range(rows):
        for c in range(cols):
            if (r, c) in wall_set:
                state[2, r, c] = 1.0
    if door_pos is not None and not door_open:
        state[3, door_pos[0], door_pos[1]] = 1.0
    if key_pos is not None and key_present and not carrying_key:
        state[4, key_pos[0], key_pos[1]] = 1.0
    state[0, agent_pos[0], agent_pos[1]] = 1.0
    state[1, goal_pos[0], goal_pos[1]] = 1.0
    if carrying_key:
        state[5, agent_pos[0], agent_pos[1]] = 1.0
    state[6 + agent_dir, agent_pos[0], agent_pos[1]] = 1.0
    return state


def build_grid_state_targets(response_text):
    response_text = normalize_reasoning_tags(response_text)
    env_type = detect_env_type(response_text)
    targets = []
    if env_type in {"minigrid", "minigrid_nav"}:
        start_pos, start_dir, goal_pos, walls, key_pos, door_pos, rows, cols = _parse_minigrid_state(response_text)
        if goal_pos is None:
            raise ValueError("Could not parse goal position from MiniGrid response text.")
        checkpoint_states = _extract_checkpoint_states(response_text)
        for pos, direction, carrying_key, key_present, door_open in checkpoint_states:
            targets.append(
                {
                    "grid": _build_minigrid_state_tensor(
                        agent_pos=pos,
                        agent_dir=direction,
                        goal_pos=goal_pos,
                        walls=walls,
                        key_pos=key_pos,
                        door_pos=door_pos,
                        carrying_key=carrying_key,
                        key_present=key_present,
                        door_open=door_open,
                        rows=rows,
                        cols=cols,
                    ),
                }
            )
    else:
        start_pos, goal_pos, holes = _parse_start_goal_holes(response_text)
        if goal_pos is None:
            raise ValueError("Could not parse goal position from response text.")
        rows = max([start_pos[0], goal_pos[0], *[h[0] for h in holes]] or [0]) + 1
        cols = max([start_pos[1], goal_pos[1], *[h[1] for h in holes]] or [0]) + 1
        checkpoint_positions = _extract_checkpoint_positions(response_text)
        for pos in checkpoint_positions:
            targets.append(
                {
                    "grid": _build_grid_state_tensor(
                        agent_pos=pos,
                        goal_pos=goal_pos,
                        holes=holes,
                        rows=rows,
                        cols=cols,
                    ),
                }
            )
    return targets


def build_textual_state_desc_response(response_text):
    """Replace each grid-token checkpoint with a natural-language [state] block."""
    response_text = normalize_reasoning_tags(response_text)
    env_type = detect_env_type(response_text)
    matches = list(GRID_TOKEN_GROUP_PATTERN.finditer(response_text))
    parts = []
    last_end = 0
    if env_type in {"minigrid", "minigrid_nav"}:
        _, _, goal_pos, walls, key_pos, door_pos, rows, cols = _parse_minigrid_state(response_text)
        if goal_pos is None:
            raise ValueError("Could not parse goal position from MiniGrid response text.")
        walls_text = "[" + ", ".join(f"({r},{c})" for r, c in sorted(walls)) + "]"
        checkpoint_states = _extract_checkpoint_states(response_text)
        if len(matches) != len(checkpoint_states):
            raise ValueError(
                f"Grid token group count {len(matches)} does not match checkpoint count {len(checkpoint_states)}."
            )
        iterator = (
            (match, state[0][0], state[0][1], state[1], state[2], state[3], state[4], walls_text, rows, cols)
            for match, state in zip(matches, checkpoint_states)
        )
        for match, agent_r, agent_c, direction, carrying_key, key_present, door_open, walls_text, rows, cols in iterator:
            parts.append(response_text[last_end:match.start()])
            if env_type == "minigrid_nav":
                parts.append(
                    "[state]\n"
                    f"agent at ({agent_r},{agent_c}) facing {list(DIR_TO_ID.keys())[direction]};\n"
                    f"goal at ({goal_pos[0]},{goal_pos[1]});\n"
                    f"walls at {walls_text};\n"
                    f"grid size is {rows}x{cols}.\n"
                    "[/state]"
                )
            else:
                key_line = (
                    "key already picked up;"
                    if carrying_key
                    else f"key at ({key_pos[0]},{key_pos[1]});" if key_pos is not None and key_present else "key no longer on map;"
                )
                door_line = (
                    f"door at ({door_pos[0]},{door_pos[1]}) is {'open' if door_open else 'locked'};"
                    if door_pos is not None
                    else "door not present;"
                )
                parts.append(
                    "[state]\n"
                    f"agent at ({agent_r},{agent_c}) facing {list(DIR_TO_ID.keys())[direction]};\n"
                    f"goal at ({goal_pos[0]},{goal_pos[1]});\n"
                    f"{key_line}\n"
                    f"{door_line}\n"
                    f"walls at {walls_text};\n"
                    f"grid size is {rows}x{cols}.\n"
                    "[/state]"
                )
            last_end = match.end()
    else:
        start_pos, goal_pos, holes = _parse_start_goal_holes(response_text)
        if goal_pos is None:
            raise ValueError("Could not parse goal position from response text.")
        rows = max([start_pos[0], goal_pos[0], *[h[0] for h in holes]] or [0]) + 1
        cols = max([start_pos[1], goal_pos[1], *[h[1] for h in holes]] or [0]) + 1
        holes_text = "[" + ", ".join(f"({r},{c})" for r, c in sorted(holes)) + "]"
        checkpoint_positions = _extract_checkpoint_positions(response_text)
        if len(matches) != len(checkpoint_positions):
            raise ValueError(
                f"Grid token group count {len(matches)} does not match checkpoint count {len(checkpoint_positions)}."
            )
        for match, (agent_r, agent_c) in zip(matches, checkpoint_positions):
            parts.append(response_text[last_end:match.start()])
            parts.append(
                "[state]\n"
                f"agent at ({agent_r},{agent_c});\n"
                f"goal at ({goal_pos[0]},{goal_pos[1]});\n"
                f"holes at {holes_text};\n"
                "safe cells are the remaining non-hole cells;\n"
                f"grid size is {rows}x{cols}.\n"
                "[/state]"
            )
            last_end = match.end()
    parts.append(response_text[last_end:])
    return "".join(parts)


def prepare_grid_target_response(response, num_grid_items, num_tokens=GRID_TOKENS_PER_IMAGE):
    """Validate and expand <grid_token> placeholders in the response."""
    response = normalize_reasoning_tags(response)
    if num_grid_items <= 0:
        return response

    group_count = count_grid_token_groups(response)
    if group_count > 0:
        if group_count != num_grid_items:
            raise ValueError(
                f"Expected {num_grid_items} <grid_token> groups, found {group_count}."
            )
        return expand_grid_token_placeholders(response, num_tokens=num_tokens)

    raise ValueError(
        f"Expected {num_grid_items} <grid_token> groups but found none in response."
    )


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def get_image_info(image, min_pixel, max_pixel, width, height):
    """Build image info dict compatible with Qwen2.5-VL processor."""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    if width and height:
        image = image.resize((width, height))
    return {
        "type": "image",
        "image": image,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel,
    }


def llava_to_openai(conversations):
    """Convert LLaVA conversation format to OpenAI-style."""
    role_mapping = {"human": "user", "gpt": "assistant"}
    result = []
    for conv in conversations:
        role = role_mapping.get(conv["from"], conv["from"])
        content = conv["value"]
        if role == "assistant":
            content = content.replace("<image>", "")
        else:
            content = content.replace("<image>", f"{GRID_VISIBLE_TOKEN}")
            # Replace <image> with proper vision token
            import re as _re
            pattern = r'\n?' + _re.escape("<image>") + r'\n?'
            from constants import VISION_START_TOKEN, VISION_END_TOKEN
            content = conv["value"]
            content = _re.sub(pattern, VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN, content)
        content = normalize_reasoning_tags(content)
        result.append({"role": role, "content": content})
    return result


# ---------------------------------------------------------------------------
# Padding utilities
# ---------------------------------------------------------------------------

def pad_sequence(sequences, padding_side='right', padding_value=0):
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    trailing_dims = sequences[0].size()[1:]
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


# ---------------------------------------------------------------------------
# DataArguments
# ---------------------------------------------------------------------------

@dataclass
class DataArguments:
    data_path: str = field(default=None)
    image_folder: Optional[str] = field(default=None)
    image_min_pixels: Optional[int] = field(default=3136)
    image_max_pixels: Optional[int] = field(default=12845056)
    image_resized_width: int = field(default=None)
    image_resized_height: int = field(default=None)
    grid_token_dropout_prob: float = field(
        default=0.0,
        metadata={"help": "Per-group dropout probability for grid token groups."},
    )
    grid_token_dropout_after_step: int = field(
        default=0,
        metadata={"help": "Start grid token dropout after this training step."},
    )
    use_grid_tokens: bool = field(
        default=True,
        metadata={"help": "Whether prompts and targets should use <grid_token> markers."},
    )
    use_textual_state_desc: bool = field(
        default=False,
        metadata={"help": "Replace grid-token checkpoints with natural-language [state] blocks."},
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GridCoTDataset(Dataset):
    """Dataset for grid CoT training with Qwen2.5-VL."""

    def __init__(self, data_path, processor, data_args, random_seed=42, shuffle=True):
        super().__init__()
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.processor = processor
        self.data_args = data_args
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.grid_token_dropout_prob = data_args.grid_token_dropout_prob
        self.grid_token_dropout_after_step = data_args.grid_token_dropout_after_step
        self.use_grid_tokens = data_args.use_grid_tokens
        self.use_textual_state_desc = data_args.use_textual_state_desc

        self.cur_step = 0
        self.rng = np.random.default_rng(seed=random_seed)

        if shuffle:
            self.rng.shuffle(self.data)

    def set_cur_step(self, step: int):
        self.cur_step = step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = self.data[i]
        processor = self.processor

        # --- Load images ---
        grid_image_paths = []
        if "images" in sample:
            all_paths = sample["images"]
            if isinstance(all_paths, str):
                all_paths = [all_paths]
            base_image_paths = [all_paths[0]]
            grid_image_paths = all_paths[1:]
            image_files = [Image.open(p).convert("RGB") for p in base_image_paths]
        else:
            image_files = []

        if not image_files:
            black = Image.new("RGB", (self.image_resized_w or 560, self.image_resized_h or 560), (0, 0, 0))
            image_files = [black]

        images = image_files

        # --- Parse conversations ---
        sources = copy.deepcopy(llava_to_openai(sample["conversations"]))
        # Prefer meta.env_type to avoid misdetection from CoT action keywords
        _meta_env = sample.get("meta", {}).get("env_type", None)
        if _meta_env in {"minigrid_nav", "minigrid", "frozenlake"}:
            env_type = _meta_env
        elif len(sources) >= 2:
            env_type = detect_env_type(sources[1].get("content", ""))
        else:
            env_type = "frozenlake"

        if env_type == "minigrid":
            if self.use_textual_state_desc:
                default_system_content = MINIGRID_SYSTEM_MESSAGE_STATE_DESC
            elif self.use_grid_tokens:
                default_system_content = MINIGRID_SYSTEM_MESSAGE
            else:
                default_system_content = MINIGRID_SYSTEM_MESSAGE_TEXT_ONLY
        elif env_type == "minigrid_nav":
            if self.use_textual_state_desc:
                default_system_content = MINIGRID_NAV_SYSTEM_MESSAGE_STATE_DESC
            elif self.use_grid_tokens:
                default_system_content = MINIGRID_NAV_SYSTEM_MESSAGE
            else:
                default_system_content = MINIGRID_NAV_SYSTEM_MESSAGE_TEXT_ONLY
        else:
            if self.use_textual_state_desc:
                default_system_content = GRID_SYSTEM_MESSAGE_STATE_DESC
            elif self.use_grid_tokens:
                default_system_content = GRID_SYSTEM_MESSAGE
            else:
                default_system_content = GRID_SYSTEM_MESSAGE_TEXT_ONLY

        if sources and sources[0]["role"] == "system":
            system_content = sources[0]["content"] or default_system_content
            sources = sources[1:]
        else:
            system_content = default_system_content

        all_input_ids, all_labels = [], []
        all_pixel_values, all_image_grid_thw = [], []
        active_grid_image_paths = list(grid_image_paths)
        grid_state_targets = []

        # System message
        sys_msg = f"{DEFAULT_IM_START_TOKEN}system\n{system_content}\n{DEFAULT_IM_END_TOKEN}\n"
        sys_ids = processor.tokenizer(sys_msg, add_special_tokens=False, return_tensors="pt")["input_ids"]
        all_input_ids.append(sys_ids.squeeze(0))
        all_labels.append(torch.full_like(sys_ids.squeeze(0), IGNORE_INDEX))

        # Process first turn only
        if len(sources) >= 2:
            user_input = sources[0]
            gpt_response = sources[1]

            # Build grid-specific prompt
            # Prefer meta rows/cols to avoid wrong inference from goal coordinates
            _meta = sample.get("meta", {})
            rows = _meta.get("rows") or None
            cols = _meta.get("cols") or None
            if rows is None or cols is None:
                rows, cols = infer_grid_size_from_text(gpt_response["content"])
            prompt_text = build_grid_user_prompt(
                rows=rows,
                cols=cols,
                include_grid_token=self.use_grid_tokens and not self.use_textual_state_desc,
                use_textual_state_desc=self.use_textual_state_desc,
                env_type=env_type,
            )

            user_content = user_input["content"]
            if DEFAULT_IMAGE_TOKEN in user_content:
                user_content = f"{user_content}\n{prompt_text}"
            else:
                user_content = f"{user_content.strip()}\n{prompt_text}" if user_content.strip() else prompt_text

            # Grid token dropout
            grid_response_content = gpt_response["content"]
            full_grid_state_targets = build_grid_state_targets(grid_response_content) if (
                self.use_grid_tokens and not self.use_textual_state_desc
            ) else []
            if (
                self.use_grid_tokens
                and not self.use_textual_state_desc
                and self.cur_step >= self.grid_token_dropout_after_step
                and self.grid_token_dropout_prob > 0
            ):
                grid_response_content, active_grid_image_paths = dropout_grid_token_groups(
                    grid_response_content, active_grid_image_paths,
                    self.grid_token_dropout_prob, self.rng,
                )
                kept_count = count_grid_token_groups(grid_response_content)
                full_grid_state_targets = full_grid_state_targets[:kept_count]
            grid_state_targets = full_grid_state_targets
            if self.use_textual_state_desc:
                grid_response_content = build_textual_state_desc_response(grid_response_content)
                active_grid_image_paths = []
                grid_state_targets = []

            user_str = (
                f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n"
                f"{user_content}\n{DEFAULT_IM_END_TOKEN}\n"
                f"{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            )
            expected_grid_groups = len(active_grid_image_paths)
            if expected_grid_groups == 0 and grid_state_targets:
                expected_grid_groups = len(grid_state_targets)
            gpt_str = f"{prepare_grid_target_response(grid_response_content, expected_grid_groups)}\n{DEFAULT_IM_END_TOKEN}\n"

            # Tokenize
            if DEFAULT_IMAGE_TOKEN in user_str:
                inputs = processor(text=[user_str], images=images, padding=False, return_tensors="pt")
                prompt_ids = inputs["input_ids"]
                all_pixel_values.append(inputs["pixel_values"])
                all_image_grid_thw.append(inputs["image_grid_thw"])
            else:
                prompt_ids = processor.tokenizer(user_str, add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"]

            response_ids = processor.tokenizer(gpt_str, add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"]

            input_ids = torch.cat([prompt_ids, response_ids], dim=1).squeeze(0)
            labels = torch.cat([
                torch.full((prompt_ids.shape[1],), IGNORE_INDEX, dtype=torch.long),
                response_ids.squeeze(0),
            ])

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        input_ids = torch.cat(all_input_ids).to(torch.long)
        labels = torch.cat(all_labels).to(torch.long)
        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if all_pixel_values:
            data_dict["pixel_values"] = torch.cat(all_pixel_values, dim=0)
            data_dict["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)

        if active_grid_image_paths:
            data_dict["grid_images"] = active_grid_image_paths
        if grid_state_targets:
            data_dict["grid_state_targets"] = grid_state_targets

        self.cur_step += 1
        return data_dict


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

class DataCollatorForGridCoT:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids, batch_labels = [], []
        batch_pixel_values, batch_image_thw = [], []
        batch_grid_images = []
        batch_grid_state_targets = []

        for ex in examples:
            batch_input_ids.append(ex["input_ids"])
            batch_labels.append(ex["labels"])
            if "pixel_values" in ex:
                batch_pixel_values.append(ex["pixel_values"])
                batch_image_thw.append(ex["image_grid_thw"])
            if "grid_images" in ex:
                batch_grid_images.append(ex["grid_images"])
            if "grid_state_targets" in ex:
                batch_grid_state_targets.append(ex["grid_state_targets"])

        input_ids = pad_sequence(batch_input_ids, padding_side="right", padding_value=self.pad_token_id)
        labels = pad_sequence(batch_labels, padding_side="right", padding_value=IGNORE_INDEX)
        attention_mask = input_ids != self.pad_token_id

        data_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        if batch_pixel_values:
            data_dict["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
            data_dict["image_grid_thw"] = torch.cat(batch_image_thw, dim=0)
        if batch_grid_images:
            data_dict["grid_images"] = batch_grid_images
        if batch_grid_state_targets:
            data_dict["grid_state_targets"] = batch_grid_state_targets

        return data_dict


def make_data_module(processor, data_args):
    dataset = GridCoTDataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
    )
    collator = DataCollatorForGridCoT(pad_token_id=processor.tokenizer.pad_token_id)
    return {"train_dataset": dataset, "eval_dataset": None, "data_collator": collator}
