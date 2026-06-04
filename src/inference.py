"""Inference utilities: constrained decoding, text cleaning, answer extraction."""

import re
from typing import List, Optional

import torch
from transformers import LogitsProcessor


# ---------------------------------------------------------------------------
# Logits processors for constrained decoding
# ---------------------------------------------------------------------------

class SuppressPadTokens(LogitsProcessor):
    """Suppress pad/anchor tokens during generation. Only allow <grid_token> (visible)."""

    def __init__(self, suppress_ids: List[int]):
        self.suppress_ids = suppress_ids

    def __call__(self, input_ids, scores):
        scores[:, self.suppress_ids] = -float("inf")
        return scores


class GridFormatEnforcer(LogitsProcessor):
    """
    State machine to enforce <think>...</think><answer>...</answer> format.
    
    States:
      expect_think_start -> inside_think -> expect_answer_start -> inside_answer -> done
    """

    def __init__(self, think_start_id, think_end_id, answer_start_id, answer_end_id, eos_id):
        self.think_start_id = think_start_id
        self.think_end_id = think_end_id
        self.answer_start_id = answer_start_id
        self.answer_end_id = answer_end_id
        self.eos_id = eos_id
        self.state = "expect_think_start"
        self._step = 0

    def __call__(self, input_ids, scores):
        self._step += 1
        
        if self._step == 1 and self.state == "expect_think_start":
            # Force <think> as first token
            mask = torch.full_like(scores, -float("inf"))
            mask[:, self.think_start_id] = scores[:, self.think_start_id]
            self.state = "inside_think"
            return mask

        last_token = input_ids[0, -1].item()

        if self.state == "inside_think":
            if last_token == self.think_end_id:
                # After </think>, force <answer>
                self.state = "expect_answer_start"
                mask = torch.full_like(scores, -float("inf"))
                mask[:, self.answer_start_id] = scores[:, self.answer_start_id]
                return mask
            # Inside think: allow anything except <answer>, </answer>, EOS
            scores[:, self.answer_start_id] = -float("inf")
            scores[:, self.answer_end_id] = -float("inf")
            scores[:, self.eos_id] = -float("inf")
            return scores

        if self.state == "expect_answer_start":
            self.state = "inside_answer"
            return scores

        if self.state == "inside_answer":
            if last_token == self.answer_end_id:
                # After </answer>, force EOS
                self.state = "done"
                mask = torch.full_like(scores, -float("inf"))
                mask[:, self.eos_id] = scores[:, self.eos_id]
                return mask
            # Inside answer: allow anything except <think>, </think>, EOS
            scores[:, self.think_start_id] = -float("inf")
            scores[:, self.think_end_id] = -float("inf")
            scores[:, self.eos_id] = -float("inf")
            return scores

        return scores


# ---------------------------------------------------------------------------
# Text cleaning and answer extraction
# ---------------------------------------------------------------------------

PATH_PATTERN = re.compile(
    r'go\s+(?:left|right|up|down)(?:\s*(?:->|=>|,|;|\||/|\\)\s*go\s+(?:left|right|up|down))*',
    flags=re.IGNORECASE,
)
ANSWER_PATTERN = re.compile(r'<answer>\s*(.*?)\s*</answer>', flags=re.IGNORECASE | re.DOTALL)


def clean_output(text: str) -> str:
    """Remove pad tokens and normalize whitespace."""
    text = text.replace("<|grid_pad|>", " ")
    text = re.sub(r'<\|[^>]+\|>', ' ', text)  # Remove any remaining special tokens
    text = text.replace('\ufffd', ' ')
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def normalize_path_text(text: str) -> Optional[str]:
    """Extract and normalize path moves from text.

    Supports both FrozenLake actions ('go left/right/up/down') and
    MiniGrid actions ('turn left', 'turn right', 'go forward',
    'pickup key', 'toggle door').
    """
    # MiniGrid / navigation actions (must come before frozenlake to avoid partial matches)
    minigrid_moves = re.findall(
        r'turn\s+(?:left|right)|go\s+forward|pickup\s+key|toggle\s+door',
        text.lower(),
    )
    if minigrid_moves:
        return ' -> '.join(m.strip() for m in minigrid_moves)

    # FrozenLake actions
    frozenlake_moves = re.findall(r'go\s+(?:left|right|up|down)', text.lower())
    if frozenlake_moves:
        return ' -> '.join(m.strip() for m in frozenlake_moves)

    return None


def extract_answer(text: str) -> Optional[str]:
    """Extract the final path answer from model output."""
    text = clean_output(text)

    # Strict mode: only accept content inside <answer>...</answer>.
    # If the tag is missing, return None so evaluators count it as no_answer.
    match = ANSWER_PATTERN.search(text)
    if match:
        answer = normalize_path_text(match.group(1))
        if answer:
            return answer

    return None


def build_suppress_token_ids(tokenizer) -> List[int]:
    """Build list of token IDs to suppress during generation."""
    suppress_strings = [
        '<|grid_pad|>',
        '<|anchor_start|>',
        '<|anchor_end|>',
        '<tool_call>',
        '</tool_call>',
    ]
    suppress_ids = []
    seen = set()
    for s in suppress_strings:
        ids = tokenizer(s, add_special_tokens=False).input_ids
        if ids and ids[0] not in seen:
            suppress_ids.append(ids[0])
            seen.add(ids[0])
    return suppress_ids
