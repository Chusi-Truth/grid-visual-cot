# 2026-03-29 Small+Near Eval Record

## Model
- /root/autodl-tmp/visual-cot_output/grid_cot_state_only_2k_run1_merged

## Eval Script
- /root/CoVT/visual-cot/scripts/eval_accuracy_by_size.py

## Shared Args
- max_new_tokens: 2048
- batch_size: 4
- threshold_area: 20
- enforce_format: False
- python: /root/miniconda3/envs/covt/bin/python

## Run 1: small_id_shift
- dataset: /root/CoVT/visual-cot/eval/2026-03-29_size_generalization_3splits/small_id_shift/grid_cot_test_small_id_shift.json
- output_dir: /root/CoVT/visual-cot/eval/2026-03-29_state_only_2k_run1_small_id_shift_tok2048
- datetime: 2026-03-29T16:29:35.123461+08:00
- total_samples: 50
- reached_goal_accuracy: 0.7000
- exact_match_accuracy: 0.1200
- status_counts: {"reached_goal": 35, "stopped_before_goal": 10, "fell_into_hole": 5}

## Run 2: near_ood
- dataset: /root/CoVT/visual-cot/eval/2026-03-29_size_generalization_3splits/near_ood/grid_cot_test_near_ood.json
- output_dir: /root/CoVT/visual-cot/eval/2026-03-29_state_only_2k_run1_near_ood_tok2048
- datetime: 2026-03-29T16:41:01.190773+08:00
- total_samples: 50
- reached_goal_accuracy: 0.2600
- exact_match_accuracy: 0.0600
- status_counts: {"stopped_before_goal": 25, "fell_into_hole": 12, "reached_goal": 13}
