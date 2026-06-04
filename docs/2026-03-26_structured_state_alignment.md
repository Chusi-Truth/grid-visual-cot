# 2026-03-26 Structured State Alignment Design

## Goal

This document records the shift from a fixed-size checkpoint target idea to a map-size-agnostic structured state alignment design for GridCoT.

The immediate goal is to move the project from:

- text-only planning with `<grid_token>` as a textual marker

to:

- structured intermediate-state supervision at each `<grid_token>` position

without assuming the map is always `4 x 5`.

## Why the Original Fixed One-Hot Idea Was Rejected

An early idea was to encode each checkpoint state as a fixed vector such as:

- agent one-hot
- goal one-hot
- hole mask
- distance scalar
- local adjacent-hole indicators

For the current `4 x 5` maps, this would have worked, but it is not scalable:

- a `4 x 5` map has 20 cells
- a larger map would immediately break a fixed 20-dimensional one-hot layout
- test-time map size changes would require redesigning the state target

Because the project may later evaluate on larger or different grid sizes, the final design must be size-agnostic.

## Final Design Choice

Instead of encoding checkpoints as a fixed 20-cell vector, each checkpoint is represented as a **multi-channel grid state tensor**:

- channel 0: agent mask
- channel 1: goal mask
- channel 2: hole mask
- channel 3: safe mask
- channel 4: normalized row coordinate map
- channel 5: normalized column coordinate map

So for a map of shape `H x W`, the target state is:

- `6 x H x W`

This preserves map structure while remaining compatible with arbitrary grid sizes.

## How the State Target Is Built

For each training sample:

1. Parse the global state from the reference CoT text:
   - start position
   - goal position
   - hole positions
2. Parse each checkpoint position following a `<grid_token>` marker.
3. Build one state tensor per checkpoint.

This means every `<grid_token>` group in the target text now has a corresponding structured state tensor.

## State Encoder

Because the raw checkpoint tensor has variable spatial size, it cannot be aligned directly with an LLM hidden state.

A small learned state encoder is used:

- `Conv2d(6 -> 32, 3x3, padding=1)`
- `SiLU`
- `Conv2d(32 -> 64, 3x3, padding=1)`
- `SiLU`
- `AdaptiveAvgPool2d(1, 1)`
- `Flatten`
- `Linear(64 -> hidden_size)`

Key property:

- `AdaptiveAvgPool2d` makes the output dimension independent of `H x W`

So the encoder maps any grid size to a fixed-dimensional embedding.

## Alignment Objective

For each visible `<grid_token>` position:

1. Take the final-layer hidden state at that token position.
2. Project it with a learned linear layer.
3. Encode the corresponding structured state tensor with the state encoder.
4. Normalize both vectors.
5. Compute alignment loss:
   - `0.5 * (MSE + cosine-distance)`

This loss is added to the normal language modeling loss:

- `total_loss = lm_loss + grid_state_loss_weight * state_alignment_loss`

## Why This Is Better Than Text-Only `<grid_token>`

Previously, `<grid_token>` mainly acted as a formatting marker.

That could encourage the model to:

- learn where to emit `<grid_token>`
- without learning the actual intermediate state

The new design instead forces the hidden state at `<grid_token>` to encode:

- where the agent is
- where the goal is
- where the holes are
- local and global spatial context

So the token becomes a true state anchor rather than a pure textual placeholder.

## Why This Is Better Than Fixed 20-Cell One-Hot

Compared with a fixed one-hot vector:

- it keeps spatial structure
- it naturally supports different map sizes
- it does not hard-code a single environment resolution
- it is closer to a true state representation

## Code Changes

### `src/data.py`

- added parsing logic for:
  - start position
  - goal position
  - holes
  - checkpoint positions after `<grid_token>`
- added `build_grid_state_targets(...)`
- added per-sample `grid_state_targets` to the dataset output
- collator now batches `grid_state_targets` as nested lists

### `src/model.py`

- added `grid_state_encoder`
- added `grid_hidden_projector`
- added `grid_state_loss_weight`
- added `_compute_grid_state_alignment_loss(...)`
- forward now computes state alignment loss when `grid_state_targets` are present

### `src/train.py`

- added `grid_state_loss_weight` to `TrainingArguments`
- passed the new loss weight into model configuration

### `scripts/train.sh`

- added `GRID_STATE_LOSS_WEIGHT` env control

## Recommended Training Start Point

Initial recommended values:

- `GRID_STATE_LOSS_WEIGHT=0.2` for conservative first runs
- `GRID_STATE_LOSS_WEIGHT=0.5` for stronger alignment once training is stable

The safest workflow is:

1. start with `0.2`
2. check whether format quality collapses
3. compare holdout `ReachedGoalAccuracy`
4. only then increase to `0.5`

## Expected Benefit

This change is intended to reduce:

- state drift during long reasoning
- wrong final steps near the goal
- hole-entry failures caused by losing local spatial awareness

The main expected indicator of improvement is:

- lower `fell_into_hole`
- higher `reached_goal`

not just better-looking CoT text.
