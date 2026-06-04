"""Token definitions, system prompts, and grid utility functions."""

import re

IGNORE_INDEX = -100

# Qwen2.5-VL chat tokens
DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

# Grid visual checkpoint tokens
GRID_PAD_TOKEN = "<|grid_pad|>"
GRID_VISIBLE_TOKEN = "<grid_token>"

# Reasoning structure tokens
THINK_START = "<think>"
THINK_END = "</think>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"

# All new tokens to register in tokenizer
NEW_TOKENS = [
    GRID_PAD_TOKEN,
    GRID_VISIBLE_TOKEN,
    THINK_START,
    THINK_END,
    ANSWER_START,
    ANSWER_END,
]

# Grid token expansion: each <grid_token> -> 1 visible + 7 pad = 8 tokens
GRID_TOKENS_PER_IMAGE = 1

GRID_SYSTEM_MESSAGE = (
    "You are an expert path planner for the FrozenLake grid environment. "
    "You are given the base grid image. Infer a safe path from S to G while avoiding all H tiles. "
    "Reason step by step inside <think>...</think>. "
    "When the reasoning reaches an important intermediate checkpoint, emit <grid_token> to mark that state. "
    "Finish with the final path inside <answer>...</answer> using actions like "
    "'go right -> go down -> ...'."
)

GRID_SYSTEM_MESSAGE_TEXT_ONLY = (
    "You are an expert path planner for the FrozenLake grid environment. "
    "You are given the base grid image. Infer a safe path from S to G while avoiding all H tiles. "
    "Reason step by step inside <think>...</think>. "
    "Finish with the final path inside <answer>...</answer> using actions like "
    "'go right -> go down -> ...'."
)

GRID_SYSTEM_MESSAGE_STATE_DESC = (
    "You are an expert path planner for the FrozenLake grid environment. "
    "You are given the base grid image. Infer a safe path from S to G while avoiding all H tiles. "
    "Reason step by step inside <think>...</think>. "
    "Use textual state summaries in [state]...[/state] when marking important intermediate checkpoints. "
    "Do not use <grid_token>. "
    "Finish with the final path inside <answer>...</answer> using actions like "
    "'go right -> go down -> ...'."
)

MINIGRID_SYSTEM_MESSAGE = (
    "You are an expert path planner for a MiniGrid-style DoorKey environment. "
    "You are given the full global map image. Infer a valid path from the agent to the goal while handling keys, locked doors, and wall tiles. "
    "Reason step by step inside <think>...</think>. "
    "When the reasoning reaches an important intermediate checkpoint, emit <grid_token> to mark that state. "
    "Finish with the final path inside <answer>...</answer> using actions like "
    "'turn left -> go forward -> pickup key -> toggle door -> ...'."
)

MINIGRID_SYSTEM_MESSAGE_TEXT_ONLY = (
    "You are an expert path planner for a MiniGrid-style DoorKey environment. "
    "You are given the full global map image. Infer a valid path from the agent to the goal while handling keys, locked doors, and wall tiles. "
    "Reason step by step inside <think>...</think>. "
    "Finish with the final path inside <answer>...</answer> using actions like "
    "'turn left -> go forward -> pickup key -> toggle door -> ...'."
)

MINIGRID_SYSTEM_MESSAGE_STATE_DESC = (
    "You are an expert path planner for a MiniGrid-style DoorKey environment. "
    "You are given the full global map image. Infer a valid path from the agent to the goal while handling keys, locked doors, and wall tiles. "
    "Reason step by step inside <think>...</think>. "
    "Use textual state summaries in [state]...[/state] when marking important intermediate checkpoints. "
    "Do not use <grid_token>. "
    "Finish with the final path inside <answer>...</answer> using actions like "
    "'turn left -> go forward -> pickup key -> toggle door -> ...'."
)

MINIGRID_NAV_SYSTEM_MESSAGE = (
    "You are an expert path planner for a MiniGrid-style navigation environment. "
    "You are given the full global map image. Infer a valid path from the agent to the goal while avoiding wall tiles. "
    "Reason step by step inside <think>...</think>. "
    "When the reasoning reaches an important intermediate checkpoint, emit <grid_token> to mark that state. "
    "Finish with the final path inside <answer>...</answer> using actions like "
    "'turn left -> go forward -> turn right -> ...'."
)

MINIGRID_NAV_SYSTEM_MESSAGE_TEXT_ONLY = (
    "You are an expert path planner for a MiniGrid-style navigation environment. "
    "You are given the full global map image. Infer a valid path from the agent to the goal while avoiding wall tiles. "
    "Reason step by step inside <think>...</think>. "
    "Finish with the final path inside <answer>...</answer> using actions like "
    "'turn left -> go forward -> turn right -> ...'."
)

MINIGRID_NAV_SYSTEM_MESSAGE_STATE_DESC = (
    "You are an expert path planner for a MiniGrid-style navigation environment. "
    "You are given the full global map image. Infer a valid path from the agent to the goal while avoiding wall tiles. "
    "Reason step by step inside <think>...</think>. "
    "Use textual state summaries in [state]...[/state] when marking important intermediate checkpoints. "
    "Do not use <grid_token>. "
    "Finish with the final path inside <answer>...</answer> using actions like "
    "'turn left -> go forward -> turn right -> ...'."
)

GRID_GOAL_PATTERN = re.compile(r"goal(?: is| =)? at \((\d+),\s*(\d+)\)", re.IGNORECASE)
GRID_COORD_PATTERN = re.compile(r"\((\d+),\s*(\d+)\)")
GRID_TOKEN_GROUP_PATTERN = re.compile(r"(?:<grid_token>)+")


def infer_grid_size_from_text(text: str):
    """Infer grid dimensions (rows, cols) from text containing goal coordinates."""
    if not text:
        return None, None
    match = GRID_GOAL_PATTERN.search(text)
    if match:
        return int(match.group(1)) + 1, int(match.group(2)) + 1
    coords = [(int(r), int(c)) for r, c in GRID_COORD_PATTERN.findall(text)]
    if not coords:
        return None, None
    return max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1


def build_grid_user_prompt(
    rows=None,
    cols=None,
    include_grid_token=True,
    use_textual_state_desc=False,
    env_type="frozenlake",
):
    """Build the user prompt for grid planning task."""
    size_text = f"The map size is {rows} rows by {cols} columns. " if rows and cols else ""
    if env_type == "minigrid":
        prompt = (
            f"This is a MiniGrid-style DoorKey map. {size_text}"
            "The image shows the full map with the agent, walls, a key, a locked door, and the goal. "
            "Find a valid path from the agent to the goal. You may need to pick up the key and toggle the door before moving through it. "
            "The action space is: turn left, turn right, go forward, pickup key, toggle door. "
            "Reason step by step in <think>...</think>. "
        )
    elif env_type == "minigrid_nav":
        prompt = (
            f"This is a MiniGrid-style navigation map. {size_text}"
            "The image shows the full map with the agent, walls, and the goal. "
            "Find a valid path from the agent to the goal while avoiding wall tiles. "
            "The action space is: turn left, turn right, go forward. "
            "Reason step by step in <think>...</think>. "
        )
    else:
        prompt = (
            f"This is a FrozenLake grid. {size_text}"
            "The image shows the full map with S=Start, F=Frozen(safe), H=Hole(danger), and G=Goal. "
            "Find a safe path from S to G while avoiding all H tiles. "
            "Reason step by step in <think>...</think>. "
        )
    if use_textual_state_desc:
        prompt += "Use [state]...[/state] textual summaries for important intermediate checkpoints. "
    elif include_grid_token:
        prompt += "Insert <grid_token> whenever you want to mark an important intermediate grid state. "
    if env_type == "minigrid":
        prompt += (
            "Finally output the chosen path in <answer>...</answer> using actions like "
            "'turn right -> go forward -> pickup key -> toggle door -> ...'."
        )
    elif env_type == "minigrid_nav":
        prompt += (
            "Finally output the chosen path in <answer>...</answer> using actions like "
            "'turn right -> go forward -> turn left -> ...'."
        )
    else:
        prompt += (
            "Finally output the chosen path in <answer>...</answer> using actions like "
            "'go right -> go down -> ...'."
        )
    return prompt
