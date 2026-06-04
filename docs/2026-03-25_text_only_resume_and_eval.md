# 2026-03-25 Text-Only Training Resume And Eval

## Summary

- Project branch: `visual-cot`
- Task setting: text-only FrozenLake planning
- Base model: `Qwen2.5-VL-7B-Instruct`
- Training data: `dataset/grid_cot_text_only.json`
- Final training artifact:
  - checkpoint: `/root/autodl-tmp/visual-cot_output/grid_cot_text_only/checkpoint-4000`
  - merged model: `/root/autodl-tmp/visual-cot_output/grid_cot_text_only_merged`
- Evaluation artifact:
  - `/root/autodl-tmp/visual-cot_output/eval_grid_cot_text_only_20260325_234319/results.json`

## What Changed

This round focused on making the text-only pipeline trainable, resumable, and easier to evaluate.

### 1. Text-only data and prompt cleanup

- Removed `<grid_token>` from the text-only dataset.
- Removed grid-token instructions from the text-only prompt path.
- Kept the target format centered on:
  - `<think> ... </think>`
  - `<answer> ... </answer>`

### 2. Data pipeline fixes

- Fixed image passing in the dataset loader so the processor receives raw images instead of message dicts.
- This resolved the earlier processor crash in `src/data.py`.

### 3. Tokenizer and embedding fixes

- Fixed tokenizer expansion in `src/train.py`.
- Fixed gradient masking so only target special-token rows are updated, using actual token ids instead of a contiguous tail-range assumption.
- This resolved the embedding hook mismatch after tokenizer resizing.

### 4. Resumable checkpoint redesign

The original checkpoint logic had two problems:

- it wrote a huge `non_lora_state_dict.bin`
- it did not reliably produce a resumable checkpoint when `save_only_model=True`

The new logic is:

- keep `embed_tokens` and `lm_head` trainable
- save only the trainable token rows during intermediate checkpoints
- store them as `trainable_token_rows.pt`
- resume training by restoring those token rows automatically
- merge final LoRA weights while also restoring the saved token rows

Files changed:

- `src/trainer.py`
- `src/train.py`
- `src/merge_lora.py`
- `scripts/train.sh`

### 5. Output directory migration

- Default training output was moved to `/root/autodl-tmp/visual-cot_output/...`
- This avoids filling the repo filesystem with checkpoints.

## Training Outcome

- Training completed to `checkpoint-4000`.
- The final checkpoint was merged successfully into:
  - `/root/autodl-tmp/visual-cot_output/grid_cot_text_only_merged`

## Quick Evaluation

Evaluation used:

- model: `/root/autodl-tmp/visual-cot_output/grid_cot_text_only_merged`
- dataset: `/root/CoVT/visual-cot/dataset/grid_cot_text_only.json`
- samples: `0, 1, 2`
- max generation length: `2048`

### Structural result

- All 3 samples produced both `<think>` and `<answer>`.
- No severe grid-token pollution was observed in these outputs.
- The previous missing-`<answer>` failure mode did not appear in this quick run.

### Task result

- Sample 0:
  - predicted path extracted successfully
  - but the path falls into hole `(3, 3)`
  - reference target is `fail to find a valid path`
- Sample 1:
  - predicted path reaches the goal
  - but does not match the reference trace
- Sample 2:
  - predicted path reaches the goal
  - but does not match the reference trace

### Quick score

- Valid path to goal: `2 / 3`
- Exact reference answer match: `0 / 3`
- Structured `<think>/<answer>` output: `3 / 3`

## Interpretation

Current model behavior is:

- structure is noticeably improved
- answer extraction is working again
- reasoning text looks fluent and task-relevant
- but planning correctness is still weak

In short:

- the model has learned the target output format
- it has not yet learned a reliable planning policy

## Main Takeaway

This run is a partial success:

- engineering stability improved substantially
- resumable training is now in place
- output structure is much better than before

But model quality is still not sufficient:

- incorrect path selection still occurs
- no-answer cases are not handled robustly
- exact path consistency with the teacher remains poor

## Next Recommended Steps

1. Run a larger evaluation slice, not just 3 samples.
2. Separate metrics into:
   - format success
   - valid path success
   - exact teacher-match success
3. Re-check whether the text-only target is over-emphasizing narration style over actual path validity.
4. Consider simplifying the reasoning target or strengthening supervision on final answer correctness.
