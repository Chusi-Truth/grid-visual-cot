# 2026-04-05 BFS Subset + Preference Notes

## State-only BFS subset results

### `state-only 500` baseline

ReachedGoalAccuracy:

- reachable100: `0.77`
- small-ood: `0.69`
- near-ood: `0.17`
- far-ood: `0.05`

### `state-only 1000`

Files:

- `eval/2026-04-05_state_only_1000_bfs_bs4_s2000_reachable100_tok4096_bs20_merged_offline`
- `eval/2026-04-05_state_only_1000_bfs_bs4_s2000_small_ood_tok4096_bs20_merged_offline`
- `eval/2026-04-05_state_only_1000_bfs_bs4_s2000_near_ood_tok4096_bs20_merged_offline`
- `eval/2026-04-05_state_only_1000_bfs_bs4_s2000_far_ood_tok4096_bs20_merged_offline`

ReachedGoalAccuracy:

- reachable100: `0.92`
- small-ood: `0.55`
- near-ood: `0.23`
- far-ood: `0.04`

Conclusion:

- Increasing BFS training subset size from `500 -> 1000` substantially improves in-domain reachable accuracy.
- OOD does not improve uniformly.
- `small-ood` degrades noticeably.
- `far-ood` remains very weak.

This suggests the larger subset helps fit the reachable distribution better, but does not solve the OOD generalization bottleneck.

## Preference experiment on top of `state-only 500 best`

New files:

- `scripts/build_preference_dataset.py`
- `scripts/train_state_only_500_preference.sh`
- `src/train_preference.py`
- `dataset/grid_cot_state_only_2k_bfs_rand500_pref.json`

Setup:

- Start from `/root/autodl-tmp/visual-cot_output/grid_cot_state_only_500_bfs_bs4_s1000_merged`
- Build `chosen/rejected` preference pairs from the `500` BFS subset
- Train a pairwise preference model for `500` steps

Observed failure:

- Debug eval on 2 reachable100 samples:
  - `eval/tmp_pref_debug_2`
  - `ReachedGoalAccuracy = 0.0`
  - `StatusCounts = {"no_answer": 2}`

Direct inspection of raw outputs shows:

- The model continues generating long reasoning text.
- The model often fails to emit a final `<answer>...</answer>` block.
- Some outputs degenerate into repeated `<|endoftext|>` tails.

Conclusion:

- This preference objective is not reliable in its current form for this task.
- The main failure mode is not just wrong planning; it is structural output collapse into `no_answer`.
- Pure pairwise ranking loss removed the strong answer-format supervision that existed in the original SFT objective.
- For this task, preference-only tuning is high-risk and should not be the default next step.

Recommendation:

- Treat this preference run as a negative result.
- Prefer narrower, more controlled experiments over replacing the original SFT objective.
