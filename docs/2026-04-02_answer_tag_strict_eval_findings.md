# 2026-04-02 Strict `<answer>` Eval Findings

## Context

- Concern: previous reachable100 scores looked inconsistent with actual outputs.
- Hypothesis: evaluator fallback (no `<answer>` tag -> still extract path from full text) inflated accuracy.

## Code Change

- Updated `src/inference.py::extract_answer` to strict behavior:
  - Only parse moves inside `<answer> ... </answer>`.
  - If tag is missing, return `None`.
  - No longest-path fallback from raw output.

## Recomputed / Rerun Results

### A) Existing old result file (post-hoc strict recompute)

- File: `eval/2026-04-02_best_old_reachable100_rerun_tok2048_bs20/results.json`
- Original recorded:
  - `ReachedGoalAccuracy = 0.42 (42/100)`
- Strict `<answer>` recompute from `raw_output`:
  - `<answer>` present: `1/100`
  - `no_answer`: `99/100`
  - `ReachedGoalAccuracy ≈ 0.00`

Conclusion: original 0.42 heavily relied on fallback extraction.

### B) Fresh strict rerun (10 samples, large token budget)

- Run: `eval/2026-04-02_best_old_reachable10_strict_tok6144_bs10`
- Config:
  - model: `grid_cot_state_only_2k_run1`
  - dataset: `grid_cot_test_reachable_100.json` (BFS reachable set)
  - `max_new_tokens = 6144`
- Result:
  - `TotalReachedGoalAccuracy = 0.0000`
  - `TotalStatusCounts = {"no_answer": 10}`

Conclusion: this is not mainly a max-token truncation issue; the core failure is missing `<answer>` structure.

## Interpretation

- Previously high scores and current strict scores are **not directly comparable**.
- Major gap source:
  1. evaluator fallback behavior (now removed),
  2. dataset definition drift (`teacher reachable` vs `BFS reachable`),
  3. weak `<answer>` formatting adherence at inference.

## Next Action

- Use strict metric as default going forward.
- When comparing models, keep fixed:
  - dataset split,
  - decoding config,
  - strict `<answer>` rule.
