# 2026-03-29 Textual State Description Ablation Training

## Goal

This ablation replaces each original `<grid_token>` checkpoint with an explicit natural-language state block:

```text
[state]
agent at (r,c);
goal at (gr,gc);
holes at [(h1,w1), (h2,w2), ...];
safe cells are the remaining non-hole cells;
grid size is HxW.
[/state]
```

The prompt is intentionally kept unchanged.

So this experiment compares:

- implicit intermediate state anchor:
  - `<grid_token>` + structured state alignment loss
- explicit textual state description:
  - natural-language `[state] ... [/state]`
  - no state alignment loss

## What Changed

The codepath now supports a `use_textual_state_desc=True` mode.

In this mode:

- the target text replaces each `<grid_token>` group with a `[state] ... [/state]` block
- the prompt is **not** changed
- `grid_state_loss_weight` is forced to `0.0`
- grid checkpoint images are not used during training

## 1. Build The Ablation Dataset

From the `visual-cot/` directory:

```bash
python scripts/build_text_state_desc_dataset.py \
  --input dataset/grid_cot_state_only_2k.json \
  --output dataset/grid_cot_text_state_desc_2k.json
```

Expected result:

- input dataset keeps the original state-only sample ordering
- output dataset keeps only the base image
- target CoT contains `[state] ... [/state]` blocks instead of `<grid_token>`

## 2. Train

Recommended launcher:

```bash
DATA_PATH=dataset/grid_cot_text_state_desc_2k.json \
OUTPUT_DIR=/root/autodl-tmp/visual-cot_output/grid_cot_text_state_desc_2k_8k \
bash scripts/train_text_state_desc_2k.sh
```

This script sets:

- `USE_TEXTUAL_STATE_DESC=True`
- `GRID_STATE_LOSS_WEIGHT=0.0`
- same base model / batch / max-step defaults as the current 2k state-only training route

Default output:

- `/root/autodl-tmp/visual-cot_output/grid_cot_text_state_desc_2k_8k`

## 3. Merge LoRA

After training:

```bash
MODEL_PATH=/root/autodl-tmp/visual-cot_output/grid_cot_text_state_desc_2k_8k \
MODEL_BASE=/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct \
SAVE_MODEL_PATH=/root/autodl-tmp/visual-cot_output/grid_cot_text_state_desc_2k_8k_merged \
bash scripts/merge_lora.sh
```

## 4. Evaluate

Batch evaluation example:

```bash
python scripts/eval_text_only_accuracy.py \
  --model-path /root/autodl-tmp/visual-cot_output/grid_cot_text_state_desc_2k_8k_merged \
  --dataset-path eval/2026-03-27_mixed_size_reachable100/grid_cot_test_mixed_size_reachable.json \
  --output-dir eval/2026-03-29_text_state_desc_2k_run1_mixed_size_reachable100 \
  --max-new-tokens 2048
```

GT>20 evaluation example:

```bash
python scripts/eval_text_only_accuracy.py \
  --model-path /root/autodl-tmp/visual-cot_output/grid_cot_text_state_desc_2k_8k_merged \
  --dataset-path eval/2026-03-28_gt20_reachable100/grid_cot_test_mixed_size_reachable.json \
  --output-dir eval/2026-03-29_text_state_desc_2k_run1_gt20_reachable100_tok4096 \
  --max-new-tokens 4096
```

## 5. Recommended Comparison Set

To make the ablation meaningful, compare these three runs under the same protocol:

1. `text-only`
2. `textual-state-description`
3. `state-only`

Recommended benchmark sets:

1. pure `4x5`
2. mixed-size
3. pure `>20`

Report at least:

- `ReachedGoalAccuracy`
- `ExactMatchAccuracy`
- `fell_into_hole`
- `out_of_bounds`
- `stopped_before_goal`

## 6. Important Note

This ablation is designed so that the **prompt stays unchanged**.

That means any gain or loss should be interpreted mainly as the effect of:

- replacing implicit hidden-state supervision

with

- explicit natural-language intermediate state description
