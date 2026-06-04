# 2026-03-30 从 Text-Only 到 State 的实验进展（详细版）

## 1. 实验目标

目标是验证从纯文本规划（text-only）到中间状态建模（state）是否能提升：

1. 路径可达率（ReachedGoalAccuracy）
2. 精确匹配率（ExactMatchAccuracy）
3. 尺度泛化（small/near_ood/gt20）

并定位失败模式（掉洞、停在终点前、输出格式漂移）。

---

## 2. 路线与设置

### 2.1 Text-Only

- 仅输入 base image
- 输出 `<think> ... </think>` + `<answer> ... </answer>`
- 不使用中间 checkpoint 图，不使用显式状态 token

### 2.2 State-Only（结构化中间状态）

- 仍仅输入 base image
- 在推理链中加入显式中间状态锚点（`<grid_token>` / state 对齐）
- 目标：强化中间规划过程

### 2.3 Textual State Description（state 文本化）

- 把中间状态表示改成 `[state] ... [/state]` 文本块
- 用于对比“符号化状态锚点”与“文本状态描述”的效果差异

---

## 3. 核心结果汇总

## 3.1 4x5 Holdout100（历史主对比）

来源：
- `eval/2026-03-27_115353_text_only_2k_run1_holdout100/summary.json`
- `eval/2026-03-27_072233_state_only_2k_run1_holdout100/summary.json`

| Run | Samples | ReachedGoal | ExactMatch | 备注 |
|---|---:|---:|---:|---|
| text_only_2k_run1 | 100 | 0.45 (45/100) | 0.02 (2/100) | `fell_into_hole` 为主 |
| state_only_2k_run1 | 100 | 0.47 (47/100) | 0.06 (6/100) | 小幅优于 text-only |

结论：在 4x5 holdout100 上，state-only 相对 text-only 有小幅收益。

## 3.2 Reachable100（可达子集）

来源：
- `eval/2026-03-27_121316_text_only_2k_run1_reachable100/summary.json`
- `eval/2026-03-27_075104_state_only_2k_run1_reachable100/summary.json`

| Run | Samples | ReachedGoal | ExactMatch |
|---|---:|---:|---:|
| text_only_2k_run1 | 100 | 0.94 | 0.19 |
| state_only_2k_run1 | 100 | 0.93 | 0.20 |

结论：在“teacher 可达筛选集”上两者接近，差异不大。

## 3.3 尺度泛化（2026-03-29，tok2048）

来源：
- `eval/2026-03-29_text_only_2k_run1_small_id_shift_tok2048/summary.json`
- `eval/2026-03-29_text_only_2k_run1_near_ood_tok2048/summary.json`
- `eval/2026-03-29_state_only_2k_run1_small_id_shift_tok2048/summary.json`
- `eval/2026-03-29_state_only_2k_run1_near_ood_tok2048/summary.json`

| Split | text-only | state-only |
|---|---:|---:|
| small_id_shift (50) | 0.66 | 0.70 |
| near_ood (50) | 0.04 | 0.26 |

结论：state-only 在 near_ood 上显著优于 text-only，是当前最明确的泛化收益证据之一。

## 3.4 Textual State Desc（run1 / run1-like）

来源：
- `eval/2026-03-29_text_state_desc_2k_8k_small_id_shift_eval/summary.json`
- `eval/2026-03-29_text_state_desc_2k_8k_near_ood_eval/summary.json`
- `eval/2026-03-30_text_state_desc_2k_8k_4x5_holdout50_eval/summary.json`

| Split | ReachedGoal | ExactMatch | 主要失败模式 |
|---|---:|---:|---|
| small_id_shift (50) | 0.40 | 0.06 | stopped_before_goal / fell_into_hole |
| near_ood (50) | 0.18 | 0.02 | stopped_before_goal 为主 |
| 4x5 holdout50 (50) | 0.36 | 0.04 | fell_into_hole + stopped_before_goal |

公平子集对比（同前 50 条）：
- text-only（前50）：`0.50`
- state-only（前50）：`0.42`
- text_state_desc（50）：`0.36`

说明：当前 text_state_desc 版本表现弱于预期，存在明显协议漂移问题（见下一节）。

---

## 4. 关键问题定位

## 4.1 训练指令与训练目标不一致（已定位）

问题：`use_textual_state_desc=True` 时，标签侧是 `[state]...[/state]`，但 system/user prompt 仍要求 `<grid_token>`。

直接后果：

1. 模型倾向输出混合格式（`[state]`、`<grid_token>`、额外噪声）
2. `<answer>` 标签闭合率下降
3. 长输出漂移，导致停在终点前和格式污染

代码定位（修复前行为）：
- `src/data.py` 里 prompt 选择依赖 `use_grid_tokens`，未优先 `use_textual_state_desc`
- `build_grid_user_prompt(... include_grid_token=...)` 在该训练脚本下传入 `True`

## 4.2 评测侧提示也可能继续污染

`eval_accuracy_by_size.py` 原先固定使用 `GRID_SYSTEM_MESSAGE`，若评测 textual-state-desc 模型，会继续要求 `<grid_token>`，加重协议偏移。

---

## 5. 已完成修复（2026-03-30）

已在代码中完成以下修复：

1. 新增 `GRID_SYSTEM_MESSAGE_STATE_DESC`  
   - `src/constants.py`
2. `build_grid_user_prompt` 在 `use_textual_state_desc=True` 时不再提示 `<grid_token>`，改为提示 `[state]...[/state]`  
   - `src/constants.py`
3. 数据管线中 `use_textual_state_desc=True` 时强制使用 state-desc system prompt，且 `include_grid_token=False`  
   - `src/data.py`
4. `train_text_state_desc_2k.sh` 默认改为 `USE_GRID_TOKENS=False`  
   - `scripts/train_text_state_desc_2k.sh`
5. `eval_accuracy_by_size.py` 新增 `--use-textual-state-desc` 开关，评测时可对齐提示协议  
   - `scripts/eval_accuracy_by_size.py`
6. 评测进度条改进：实时显示 `processed/reached_acc/exact_acc`，去除逐步刷屏打印  
   - `scripts/eval_accuracy_by_size.py`

---

## 6. 当前判断（阶段性）

1. state-only 路线是有效的，尤其在 near_ood 上相对 text-only 有明显增益。
2. textual-state-desc 目前的低表现更像“实现/协议问题”而非“思路本身无效”。
3. 下一步应优先看修复后 run2 是否回升，再决定是否继续扩大 textual-state-desc 训练规模。

---

## 7. 建议的下一步验证

1. 使用修复后的脚本重训 `text_state_desc_run2`（2k->8k）
2. 用同一评测口径重跑：
   - 4x5 holdout50
   - small_id_shift (50)
   - near_ood (50)
3. 与以下基线做同口径 A/B：
   - text-only_2k_run1
   - state-only_2k_run1
4. 输出三份可复现材料：
   - `summary.json`
   - `results.json`
   - size heatmap（按 rows x cols）

