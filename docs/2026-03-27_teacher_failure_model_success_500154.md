# 2026-03-27 Teacher-Failure / Model-Success Case: `500154.jpg`

## Why This Case Matters

This sample is important because it shows a nontrivial phenomenon:

- the teacher reference failed to produce a valid path
- but the trained model did produce a valid path to the goal

So this is evidence that the current model is not merely copying the teacher trajectory distribution.

At least on this sample, the model found a better solution than the stored teacher answer.

## Sample Identity

- image:
  - `/root/CoTCreator/eval/bagel_plan/cot_visual_500/base_images/500154.jpg`
- source dataset:
  - `eval/2026-03-27_reachable_holdout100/grid_cot_candidate_140.json`
- sample index in candidate set:
  - `127`
- single-sample eval output:
  - `eval/2026-03-27_single_500154/results.json`

## Environment Description

The sample states:

- start: `(0, 0)`
- goal: `(3, 4)`
- holes:
  - `(0,3)`
  - `(1,0)`
  - `(1,1)`
  - `(2,3)`
  - `(3,2)`

## Teacher Reference Behavior

The teacher CoT searches for a path but eventually degenerates into repeated local looping around `(3,1)`.

The final teacher answer is:

```text
<answer>fail to find a valid path</answer>
```

This is notable because the map is in fact solvable.

So the teacher here is not an oracle.

## Model Prediction

Model used:

- `/root/autodl-tmp/visual-cot_output/grid_cot_state_only_2k_run1_merged`

Predicted answer:

```text
go right -> go right -> go down -> go down -> go up -> go right -> go right -> go down -> go down
```

The raw output was saved in:

- `eval/2026-03-27_single_500154/results.json`

## Why the Model Answer Is Better

The predicted path avoids all listed holes and reaches the goal:

1. `(0,0)` -> right -> `(0,1)`
2. `(0,1)` -> right -> `(0,2)`
3. `(0,2)` -> down -> `(1,2)`
4. `(1,2)` -> down -> `(2,2)`
5. `(2,2)` -> up -> `(1,2)`
6. `(1,2)` -> right -> `(1,3)`
7. `(1,3)` -> right -> `(1,4)`
8. `(1,4)` -> down -> `(2,4)`
9. `(2,4)` -> down -> `(3,4)`

Final state:

- goal reached

So on this sample:

- teacher label: failure
- model output: success

## Interpretation

This case supports several important conclusions:

1. Teacher data is not always optimal.

- the current dataset can contain teacher failures even on solvable maps

2. The model is not purely memorizing teacher answers.

- if it were only copying the teacher target behavior, this sample should also fail

3. Evaluation against teacher exact match is not sufficient.

- exact match would mark this sample as incorrect
- but environment execution shows that the model is actually better than the teacher

4. Environment-based evaluation is essential.

- executable rollout is a more trustworthy signal than teacher-string agreement

## What This Case Does and Does Not Prove

This case is strong evidence for the following statement:

- on at least some samples, the trained model already exceeds the teacher search heuristic

This is a meaningful result because:

- the image is not present in the current training sets
- the teacher fails on a solvable map
- the model succeeds on that same map

So the model is not merely reproducing the teacher target verbatim.

However, this case alone does **not** prove the stronger claim that:

- the model has already surpassed the teacher overall

To support that stronger claim, the project still needs a batch-level comparison over many samples, for example:

- teacher success / model success
- teacher success / model fail
- teacher fail / model success
- teacher fail / model fail

So the current safest conclusion is:

- the model has demonstrated teacher-surpassing behavior on at least a subset of samples
- but overall superiority over the teacher still needs systematic measurement

## Implication for Future Work

This case strengthens the motivation for:

- executable evaluation
- bootstrap self-improvement
- preference construction based on environment success
- caution when using teacher exact-match as the main metric

In particular, it suggests that a future bootstrap + DPO pipeline could be meaningful because:

- the current model may already produce trajectories that improve upon the original teacher search in some cases

## Recommended Follow-Up

Search systematically for more cases of:

- teacher failure + model success

If these cases are not rare, that would imply:

- the training labels are partially suboptimal
- the model has already begun to exceed the teacher on some subset of the data
