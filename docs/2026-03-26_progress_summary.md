# 2026-03-26 Progress Summary

## Scope

This document summarizes the recent progress on the `visual-cot` branch after the earlier text-only baseline work.

The main themes in this round were:

- restoring `<grid_token>` into the text-only CoT targets
- building proper evaluation tools
- measuring holdout performance on a separate 100-sample test set
- moving from plain `<grid_token>` formatting toward structured intermediate-state supervision

## 1. Text-Only Data Was Reconfigured

The current text-only training dataset is still:

- base-image-only as model input

But it no longer removes `<grid_token>` from the target CoT.

Current setup:

- dataset file: `dataset/grid_cot_text_only.json`
- images per sample: only the base image is kept
- assistant target: retains `<grid_token>` markers inside `<think> ... </think>`

This means the branch is no longer pure text-only formatting.

It is now better described as:

- base-image input
- text CoT target
- explicit intermediate `<grid_token>` markers

## 2. Prompting Was Updated

`src/constants.py` was updated so both:

- system prompt
- user prompt

explicitly instruct the model to emit `<grid_token>` at important intermediate states.

This keeps the target distribution and prompt instructions aligned.

## 3. Simulation / Evaluation Utilities Were Added

Two new utility scripts were added.

### `scripts/simulate_grid_path.py`

Purpose:

- take one predicted answer
- replay the move sequence
- report whether it:
  - reaches the goal
  - falls into a hole
  - goes out of bounds
  - stops before the goal

This was used to verify that:

- sample 0 from the previous quick evaluation was wrong
- samples 1 and 2 were valid goal-reaching paths

### `scripts/eval_text_only_accuracy.py`

Purpose:

- run batch evaluation over a dataset
- extract final answers
- simulate them against the reference environment
- compute summary metrics

Metrics reported include:

- reached-goal accuracy
- exact-match accuracy
- failure-type counts

## 4. A Separate 100-Sample Holdout Test Set Was Created

A new holdout set was generated under:

- `eval/2026-03-26_062405_text_only_holdout100`

Properties:

- 100 samples
- generated separately from the training data
- includes base images and teacher CoT / answer references

Teacher distribution in the holdout set:

- 83 reachable-path cases
- 17 no-path cases

## 5. Holdout Evaluation Result

Evaluated model:

- `/root/autodl-tmp/visual-cot_output/grid_cot_text_only_merged`

Evaluation output directory:

- `eval/2026-03-26_062405_text_only_holdout100/text_only_eval_2026-03-26_142621`

Key result:

- `ReachedGoalAccuracy = 0.2800`
- `ExactMatchAccuracy = 0.0000`

Counts:

- reached goal: 28
- fell into hole: 48
- stopped before goal: 12
- out of bounds: 12
- no answer: 0

Interpretation:

- the model can usually emit a path string
- but planning quality is still poor
- the main failure mode is entering holes

So the current model has learned:

- output formatting
- path-like language generation

but has not yet learned a robust planning policy.

## 6. Conceptual Clarification on `<grid_token>`

The project direction was clarified during this round:

- `<grid_token>` should be treated as an interface slot
- not as a fixed-format token with a single meaning

It can supervise:

- pure formatting
- environment state representation
- visual representation
- or a joint representation

This led to a key design decision:

- for FrozenLake, direct structured state supervision is more appropriate than image-feature supervision as the next step

## 7. Structured State Alignment Was Added

Instead of supervising `<grid_token>` with a fixed-size 20-cell one-hot representation, the implementation was changed to a size-agnostic structured state representation.

### Data-side representation

For each checkpoint, the pipeline now builds a variable-size multi-channel state tensor:

- agent mask
- goal mask
- hole mask
- safe mask
- normalized row coordinates
- normalized column coordinates

Shape:

- `6 x H x W`

This design works even if map size changes at test time.

### Model-side alignment

The model now includes:

- a small grid-state encoder:
  - Conv
  - Conv
  - AdaptiveAvgPool
  - Linear
- a projector from `<grid_token>` hidden state into the same hidden space

Loss:

- state alignment loss between:
  - `<grid_token>` hidden state
  - encoded structured checkpoint state

Current form:

- `0.5 * (MSE + cosine-distance)`

Total training loss:

- `language modeling loss + grid_state_loss_weight * state_alignment_loss`

## 8. Why the Structured State Route Was Chosen

This design was preferred over direct checkpoint-image encoding because FrozenLake checkpoints are fundamentally symbolic environment states.

The image is only a rendering of:

- agent position
- goal position
- hole layout
- safe cells

So supervising `<grid_token>` with:

- structured environment state

is cleaner than supervising it with:

- raw image appearance

At the same time, this design remains extensible to larger maps because it does not hard-code a fixed cell count.

## 9. Main Current Status

The branch is now at the following stage:

- base image as input
- target CoT includes `<grid_token>`
- `<grid_token>` is no longer only a formatting marker
- `<grid_token>` positions are now intended to encode environment state

This is not yet full image-based visual alignment.

It is better described as:

- text generation
- plus structured intermediate-state alignment

## 10. Recommended Next Step

The next practical experiment is:

- increase training data beyond 500 samples
- keep base-image-only input
- retain `<grid_token>` plus intermediate-state text in CoT targets
- continue evaluating on the separate 100-sample holdout set

## 11. New 1k-State-Only Experiment

After the earlier 500-sample runs, a larger state-only dataset was built:

- dataset: `dataset/grid_cot_state_only_1k.json`
- sample count: 1000
- input images: base image only
- target CoT: keeps `<grid_token>` and intermediate state narration
- checkpoint images: removed from the training samples

Supporting script added:

- `scripts/train_state_only_1k.sh`

This script trains the same state-alignment route on the 1k dataset and uses:

- default `MAX_STEPS=8000`
- default `SAVE_STEPS=2000` in the underlying train entry unless overridden
- base model `/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct`

## 12. Dataset Generation Bug Was Fixed

While building the 1k dataset, one bug surfaced:

- generated samples could reference base images with non-contiguous indices
- example missing file: `base_images/1158.jpg`

Cause:

- `build_dataset.py` uses `episode_used` as the image index
- rejected unreachable maps create index gaps

Fix:

- `scripts/build_dataset.py` now ensures the base image exists
- if the referenced base image is missing, it renders and saves it on the fly

This prevents future training failures caused by missing base-image files.

## 13. Quick 3-Sample Evaluation of the New 1k/8k Model

Merged model:

- `/root/autodl-tmp/visual-cot_output/grid_cot_state_only_1k_8k_merged`

Quick evaluation output:

- `eval/2026-03-26_132648_state_only_1k_8k_quick3`

Result:

- sample 0: reached goal
- sample 1: reached goal
- sample 2: reached goal

Observed generation quality:

- `<think>` / `</think>` structure is complete
- `<answer>` is present in all 3 cases
- `<grid_token>` appears inside the reasoning trace
- no obvious token-garbling was observed in these samples

## 14. Holdout-100 Result for the New 1k/8k Model

Evaluation model:

- `/root/autodl-tmp/visual-cot_output/grid_cot_state_only_1k_8k_merged`

Evaluation dataset:

- `eval/2026-03-26_062405_text_only_holdout100/grid_cot_test_100.json`

Evaluation output:

- `eval/2026-03-26_150501_state_only_1k_8k_holdout100`

Key metrics:

- `ReachedGoalAccuracy = 0.4700`
- `ExactMatchAccuracy = 0.1100`
- `ReachedGoalCount = 47 / 100`
- `NoAnswerCount = 0`

Failure breakdown:

- `fell_into_hole = 39`
- `out_of_bounds = 14`

Interpretation:

- this is a large improvement over the earlier 500-sample state-align run
- the model now reliably emits complete answers
- the dominant remaining failures are no longer formatting failures
- the dominant remaining failures are unsafe plans:
  - stepping into holes
  - going out of bounds

## 15. Current Conclusion

This round supports three conclusions:

- data scale was a real bottleneck at 500 samples
- increasing data and training steps materially improved holdout performance
- the remaining problem is now more about planning safety than output structure

In short:

- the system is no longer mainly failing because it cannot format answers
- it is now mainly failing because some generated action sequences are still unsafe

1. train the new state-alignment version
2. compare against the previous text-only run on the same holdout set
3. focus on whether:
   - `fell_into_hole` decreases
   - `reached_goal` increases

That comparison is the first real test of whether `<grid_token>` is starting to act as a meaningful intermediate state anchor.
