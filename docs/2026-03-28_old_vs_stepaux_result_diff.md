# 2026-03-28 老版 vs 新版（step-aux）实验结果差异

## 1. 对比对象

- 老版 state-only：
  - model: `/root/autodl-tmp/visual-cot_output/grid_cot_state_only_2k_run1_merged`
- 新版 state-only（step-aux）：
  - model: `/root/autodl-tmp/visual-cot_output/grid_cot_state_only_2k_with_stepaux_run1`
  - 相比老版，`grid_token` 对齐目标新增 3 维步数辅助特征：
    - `min_total_steps_lb`
    - `remaining_steps_lb`
    - `steps_used_lb`

---

## 2. 核心结果快照

### 2.1 纯 4x5 测试集

- 老版（100样本）
  - 结果文件：
    - `eval/2026-03-27_075104_state_only_2k_run1_reachable100/summary.txt`
  - 指标：
    - `ReachedGoalAccuracy = 0.9300`（93/100）
    - `ExactMatchAccuracy = 0.2000`

- 新版（10样本快速评测）
  - 结果文件：
    - `eval/2026-03-28_state_only_stepaux_run1_4x5_reachable10_tok1024/summary.txt`
  - 指标：
    - `ReachedGoalAccuracy = 0.7000`（7/10）
    - `ExactMatchAccuracy = 0.2000`

### 2.2 纯 >20 测试集

- 老版（100样本, tok4096）
  - 结果文件：
    - `eval/2026-03-28_state_only_2k_run1_gt20_reachable100_tok4096/summary.txt`
  - 指标：
    - `ReachedGoalAccuracy = 0.0700`（7/100）
    - `ExactMatchAccuracy = 0.0000`

- 新版（10样本快速评测）
  - tok1024：
    - `eval/2026-03-28_state_only_stepaux_run1_gt20_reachable10_tok1024/summary.txt`
    - `ReachedGoalAccuracy = 0.0000`（0/10）
  - tok3072：
    - `eval/2026-03-28_state_only_stepaux_run1_gt20_reachable10_tok3072/summary.txt`
    - `ReachedGoalAccuracy = 0.3000`（3/10）
  - tok6144：
    - `eval/2026-03-28_state_only_stepaux_run1_gt20_reachable10_tok6144/summary.txt`
    - `ReachedGoalAccuracy = 0.3000`（3/10）

---

## 3. 结果差异解读（当前阶段）

1. 新版在 >20 上对 `max_new_tokens` 高度敏感：
- 1024 -> 3072 有明显提升（0% -> 30% in 10-sample quick check）
- 3072 -> 6144 无进一步提升（仍 30%）

2. 当前观测到的新旧差异，不能直接归因为“3维 step-aux 一定更好/更差”，原因：
- 样本规模不一致（老版常用100，新版当前多为10样本快测）
- 推理链路不完全一致（新版大量使用运行时 merge）
- 输出结构稳定性问题仍在（`<answer>` 标签不稳定）

3. 目前可以确定的事实：
- 新版引入 step-aux 后，仍然存在 >20 的规划瓶颈（掉洞 + 提前停）
- 新版在少样本快测里可达到一定 >20 成功率（3/10），但统计不稳定

---

## 4. 下一步对比建议（保证公平）

如需得出“老版 vs 新版”有效结论，建议按以下固定协议复测：

1. 同一测试集规模（建议都用 100 样本）
2. 同一 token 上限（建议 4096）
3. 同一推理链路（都用 merged 或都用 runtime-merge）
4. 同一边界语义（FrozenLake 越界=原地不动）
5. 同时报告：
- `ReachedGoalAccuracy`
- `fell_into_hole / stopped_before_goal`
- `<answer>` 输出完整率

