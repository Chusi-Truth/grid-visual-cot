# 2026-03-26 State-Only 1k / 8k Experiment

## Setup

- model family: Qwen2.5-VL-7B-Instruct + LoRA
- training data: `dataset/grid_cot_state_only_1k.json`
- sample count: 1000
- input modality: base image only
- target CoT:
  - keep `<think> ... </think>`
  - keep `<answer> ... </answer>`
  - keep `<grid_token>`
  - keep intermediate state narration
- state supervision:
  - structured state alignment on `<grid_token>`

## Training Configuration

- script: `scripts/train_state_only_1k.sh`
- recommended default:
  - `MAX_STEPS=8000`
  - `SAVE_STEPS=4000`
- trained output:
  - `/root/autodl-tmp/visual-cot_output/grid_cot_state_only_1k_8k`
- merged model:
  - `/root/autodl-tmp/visual-cot_output/grid_cot_state_only_1k_8k_merged`

## Quick 3-Sample Check

Evaluation directory:

- `eval/2026-03-26_132648_state_only_1k_8k_quick3`

Result:

- sample 0: reached goal
- sample 1: reached goal
- sample 2: reached goal

Observed output quality:

- complete `<think>` sections
- complete `<answer>` sections
- `<grid_token>` appears during reasoning
- no obvious garbled token pollution in the checked samples

## Holdout-100 Result

Evaluation directory:

- `eval/2026-03-26_150501_state_only_1k_8k_holdout100`

Dataset:

- `eval/2026-03-26_062405_text_only_holdout100/grid_cot_test_100.json`

Metrics:

- `ReachedGoalAccuracy: 0.4700`
- `ExactMatchAccuracy: 0.1100`
- `ReachedGoalCount: 47`
- `ExactMatchCount: 11`
- `NoAnswerCount: 0`

Failure breakdown:

- `fell_into_hole: 39`
- `out_of_bounds: 14`

## Comparison With Earlier State-Align Run

Earlier holdout result:

- around `0.21` reached-goal accuracy on holdout 100

Current result:

- `0.47` reached-goal accuracy on holdout 100

Interpretation:

- scaling from 500 to 1000 samples and training longer helped substantially
- output completeness is much better than before
- the main remaining weakness is plan safety, not output formatting

## Current Conclusion

This experiment suggests:

- data scale was one real bottleneck
- but the remaining bottleneck is not only data quantity
- the model now usually produces full answers
- remaining failures mostly come from unsafe action choices

So the next improvement target should focus on:

- reducing hole entries
- reducing out-of-bounds moves
- strengthening path-safety supervision
