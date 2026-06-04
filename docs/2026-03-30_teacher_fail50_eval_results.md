# 2026-03-30 Teacher-Fail Reachable-50 Eval Results

## Dataset
- Path: `/root/CoVT/visual-cot/eval/2026-03-30_teacher_fail_but_reachable50_regen/grid_cot_test_teacher_fail_but_reachable_50.json`
- Construction:
  - Base 18 samples from: `eval/2026-03-30_teacher_fail_but_reachable30_bfs/grid_cot_test_teacher_fail_but_reachable_30.json`
  - Newly generated 32 samples with rule: `BFS reachable + weak teacher beam-search failed`
- Validation:
  - Total samples: `50`
  - Teacher fail labels: `50/50`
  - BFS reachable: `50/50`
  - Missing images: `0`

## Eval Settings
- Script: `scripts/eval_accuracy_by_size.py`
- Shared settings:
  - `max_new_tokens=1536`
  - `batch_size=4`
  - `enforce-format=False`
  - `threshold_area=20`

## Results (50 samples)
- text-only
  - Model: `/root/autodl-tmp/visual-cot_output/grid_cot_text_only_2k_run1`
  - ReachedGoalAccuracy: `0.2000` (`10/50`)
  - ExactMatchAccuracy: `0.0000`
  - StatusCounts: `{"fell_into_hole": 15, "stopped_before_goal": 25, "reached_goal": 10}`
  - Output: `/root/CoVT/visual-cot/eval/2026-03-30_teacher_fail50_text_only_eval_tok1536`

- state-only
  - Model: `/root/autodl-tmp/visual-cot_output/grid_cot_state_only_2k_with_stepaux_run1`
  - ReachedGoalAccuracy: `0.2600` (`13/50`)
  - ExactMatchAccuracy: `0.0000`
  - StatusCounts: `{"stopped_before_goal": 18, "fell_into_hole": 19, "reached_goal": 13}`
  - Output: `/root/CoVT/visual-cot/eval/2026-03-30_teacher_fail50_state_eval_tok1536`

- state-desc
  - Model: `/root/autodl-tmp/visual-cot_output/grid_cot_text_state_desc_2k_8k_run2`
  - ReachedGoalAccuracy: `0.2400` (`12/50`)
  - ExactMatchAccuracy: `0.0000`
  - StatusCounts: `{"stopped_before_goal": 35, "reached_goal": 12, "fell_into_hole": 3}`
  - Output: `/root/CoVT/visual-cot/eval/2026-03-30_teacher_fail50_state_desc_eval_tok1536`

## Quick Conclusion
- On this teacher-fail reachable subset:
  - `state-only (0.26)` > `state-desc (0.24)` > `text-only (0.20)`
- The gain is mainly in reached-goal rate; exact-match remains `0.0` for all three.
