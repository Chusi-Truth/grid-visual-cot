# 2026-03-31 Far-OOD 与 Size 总览实验总结

## 本轮目标
- 在已有 `small_id_shift`、`near_ood` 基础上补齐 `far_ood`，并做三模型统一对比。
- 指标统一使用 `ReachedGoalAccuracy`（是否到达终点）。

## 评测设置
- `far_ood` 数据集：`eval/2026-03-29_size_generalization_3splits/far_ood/grid_cot_test_far_ood.json`
- 样本数：40
- 模型：
  - text-only: `grid_cot_text_only_2k_run1_merged`
  - state-only: `grid_cot_state_only_2k_run1_merged`
  - state-desc: `grid_cot_text_state_desc_2k_8k_run2`
- 解码长度：`max_new_tokens=4096`（far_ood）

## Far-OOD 结果
- text-only: `0.0000`（0/40）
- state-only: `0.0500`（2/40）
- state-desc: `0.0500`（2/40）

结果目录：
- `eval/2026-03-31_text_only_2k_run1_far_ood_tok4096_bs4`
- `eval/2026-03-31_state_only_2k_run1_far_ood_tok4096`
- `eval/2026-03-31_text_state_desc_2k_8k_run2_far_ood_tok4096_bs4`

## small/near/far 总览（ReachedGoalAccuracy）
- small_id_shift (n=50): text `0.66` / state `0.70` / state-desc `0.50`
- near_ood (n=50): text `0.04` / state `0.26` / state-desc `0.22`
- far_ood (n=40): text `0.00` / state `0.05` / state-desc `0.05`

总览热力图目录：
- `eval/2026-03-31_size_heatmaps_text_state_stateDesc_small_near_far`
- 图文件：`heatmap_small_near_far_reached_goal_acc.png`

## 结论（当前阶段）
- 在更强 OOD（far_ood）上，三模型都明显退化，当前泛化瓶颈仍然显著。
- state 系列（state/state-desc）相比 text-only 仍有小幅优势，但绝对值仍低。
- 下一步重点应放在：降低 `stopped_before_goal` 和 `fell_into_hole`，而不是只追求格式输出。
