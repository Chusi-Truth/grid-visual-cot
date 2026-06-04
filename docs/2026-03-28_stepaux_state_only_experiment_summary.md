# 2026-03-28 Step-Aux State-Only 实验总结

## 1. 本轮做了什么

1. 在 `state-only` 路线中加入了与步数相关的 3 维辅助状态特征（step-aux）：
- `min_total_steps_lb`
- `remaining_steps_lb`
- `steps_used_lb`

2. 训练得到新模型目录：
- `/root/autodl-tmp/visual-cot_output/grid_cot_state_only_2k_with_stepaux_run1`

3. 由于数据盘空间紧张（无法稳定产出新的 16G merged 目录），评测采用了运行时加载：
- base model + LoRA + `non_lora_state_dict.bin` in-memory 合并推理

4. 评测设置使用了两类测试集：
- 纯 4x5：`eval/2026-03-27_reachable_holdout100/grid_cot_test_reachable_100.json`
- 纯 >20：`eval/2026-03-28_gt20_reachable100/grid_cot_test_mixed_size_reachable.json`

---

## 2. 关键评测结果（新模型）

### 2.1 纯 4x5（快速 10 样本）
- 输出目录：
  - `eval/2026-03-28_state_only_stepaux_run1_4x5_reachable10_tok1024`
- 指标：
  - `ReachedGoalAccuracy = 0.7000`（7/10）
  - `ExactMatchAccuracy = 0.2000`
  - 失败主要为 `stopped_before_goal`

### 2.2 纯 >20（快速 10 样本）

1) `max_new_tokens=1024`
- 输出目录：
  - `eval/2026-03-28_state_only_stepaux_run1_gt20_reachable10_tok1024`
- 指标：
  - `ReachedGoalAccuracy = 0.0000`（0/10）

2) `max_new_tokens=3072`
- 输出目录：
  - `eval/2026-03-28_state_only_stepaux_run1_gt20_reachable10_tok3072`
- 指标：
  - `ReachedGoalAccuracy = 0.3000`（3/10）
  - `ExactMatchAccuracy = 0.0000`

3) `max_new_tokens=6144`
- 输出目录：
  - `eval/2026-03-28_state_only_stepaux_run1_gt20_reachable10_tok6144`
- 指标：
  - `ReachedGoalAccuracy = 0.3000`（3/10）
  - `ExactMatchAccuracy = 0.0000`

---

## 3. 观察与结论

1. `>20` 上 token 上限影响明显：
- 1024 -> 3072 有显著提升（0% -> 30% in 10-sample quick check）。
- 3072 -> 6144 没有继续提升（仍 30%）。

2. 当前瓶颈不只在 token 长度：
- 虽然路径可被抽取，结构化 `<answer>...</answer>` 输出仍不稳定。
- 这导致评测中大量样本依赖 fallback 路径抽取，而非规范最终答案。

3. 失败类型在 >20 仍集中于：
- `fell_into_hole`
- `stopped_before_goal`

---

## 4. 与旧模型对照（摘要）

旧模型（`grid_cot_state_only_2k_run1`）在 >20 的 100 样本曾得到：
- `ReachedGoalAccuracy = 0.07`（tok4096）

新模型当前仅跑了 >20 的 10 样本快速评测：
- 最优为 `0.30`（tok3072/6144）

注意：两者样本规模不同（100 vs 10），不能直接下最终优劣结论；需要同协议同样本规模复测。

---

## 5. 下一步建议

1. 固定协议做正式复测：
- 同一测试集规模（建议 100）  
- 固定 token 上限（建议 4096）  
- 固定边界语义（越界原地不动）

2. 优先解决结构输出稳定性：
- 强化 `<think>/<answer>` 输出约束（训练或解码侧）
- 再观察去除格式噪声后的真实规划能力增益

