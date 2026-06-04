# MiniGrid Navigation 实验记录

---

## Run 1 — text_only baseline

| 项目 | 值 |
|---|---|
| **output_dir** | `minigrid4x4_nav_text_only_2000_s8000_run1` |
| **dataset** | `minigrid4x4_nav_text_only_2000.json` (2000 samples) |
| **CoT 格式** | 旧版：逐步动作对比，无累积路径前缀 |
| **use_grid_tokens** | False |
| **max_steps** | 4000 |
| **lora_rank / alpha** | 16 / 32 |
| **lora_dropout** | 0.05 |
| **learning_rate** | 5e-6 |
| **weight_decay** | 0.05 |
| **answer_token_loss_weight** | 10.0 |
| **answer_span_loss_weight** | 20.0 |
| **answer_transition_loss_weight** | 15.0 |
| **grid_state_loss_weight** | 0.0 |
| **freeze_vision_tower** | True |
| **freeze_llm** | True |
| **loss: start → end** | 3.80 → 0.010 |
| **训练集准确率** | 0/3 (0%) |
| **问题** | 模型幻觉 goal/walls；CoT 格式正确但规划错误；loss 收敛过快（step ~500 即 <0.2） |

---

## Run 2 — state_only + grid_state_loss=1.0

| 项目 | 值 |
|---|---|
| **output_dir** | `minigrid4x4_nav_state_only_2000_s8000_run1` |
| **dataset** | `minigrid4x4_nav_state_only_2000.json` (2000 samples, 有 checkpoint 图) |
| **CoT 格式** | 旧版：有 `<grid_token>`，每个对应独立 checkpoint 图 |
| **use_grid_tokens** | True |
| **max_steps** | 4000 |
| **lora_rank / alpha** | 16 / 32 |
| **lora_dropout** | 0.05 |
| **learning_rate** | 5e-6 |
| **weight_decay** | 0.05 |
| **answer_token_loss_weight** | 10.0 |
| **answer_span_loss_weight** | 20.0 |
| **answer_transition_loss_weight** | 15.0 |
| **grid_state_loss_weight** | 1.0 |
| **freeze_vision_tower** | True |
| **freeze_llm** | True |
| **loss: start → end** | 14.35 → 0.058 |
| **训练集准确率** | 0/3 (0%)，no_answer=3 |
| **问题** | enforce_format + grid_token 冲突导致无限重复循环；eval 只传 base image 但训练用多张 checkpoint 图，训练/推理不一致 |

---

## Run 3 — v2 CoT + base_image_only + grid_state_loss=0.0

| 项目 | 值 |
|---|---|
| **output_dir** | `minigrid4x4_nav_v2_2000_run1` |
| **dataset** | `minigrid4x4_nav_v2_2000.json` (2000 samples) |
| **CoT 格式** | 新版 v2：FrozenLake 风格累积路径前缀；images 含 base + checkpoint 图（有误） |
| **use_grid_tokens** | True |
| **max_steps** | 4000 |
| **lora_rank / alpha** | 16 / 32 |
| **lora_dropout** | 0.05 |
| **learning_rate** | 5e-6 |
| **weight_decay** | 0.05 |
| **answer_token_loss_weight** | 10.0 |
| **answer_span_loss_weight** | 20.0 |
| **answer_transition_loss_weight** | 15.0 |
| **grid_state_loss_weight** | 0.0 |
| **freeze_vision_tower** | True |
| **freeze_llm** | True |
| **图像风格** | 旧版 MiniGrid 渲染（黑底灰墙，对比度低）|
| **loss: start → end** | 4.02 → 0.013 |
| **训练集准确率** | 0/3 (0%) |
| **问题** | 训练用多张图、推理只传 base image，训练/推理不一致；grid_state_loss=0 无监督；图像对比度低 |

---

## Run 4 — v2 CoT + 新图像风格 + base_image_only + grid_state_loss=1.0（待跑）

| 项目 | 值 |
|---|---|
| **output_dir** | `minigrid4x4_nav_v2_2000_run2` |
| **dataset** | `minigrid4x4_nav_v2_2000.json` (重新生成，只有 base image) |
| **CoT 格式** | 新版 v2：FrozenLake 风格累积路径；images 只有 base image（与 FrozenLake 一致） |
| **use_grid_tokens** | True |
| **max_steps** | 4000 |
| **lora_rank / alpha** | 16 / 32 |
| **lora_dropout** | 0.05 |
| **learning_rate** | 5e-6 |
| **weight_decay** | 0.05 |
| **answer_token_loss_weight** | 10.0 |
| **answer_span_loss_weight** | 20.0 |
| **answer_transition_loss_weight** | 15.0 |
| **grid_state_loss_weight** | 1.0 |
| **freeze_vision_tower** | True |
| **freeze_llm** | True |
| **图像风格** | 新版 PIL 渲染：深灰背景 + 橙色波浪障碍 + 亮绿 goal + 红色箭头 agent |
| **loss: start → end** | TBD |
| **训练集准确率** | TBD |
| **改进点** | 修复训练/推理图像不一致；开启 grid_state_loss；高对比度图像 |

---

## 关键结论

1. **CoT 格式**：旧版逐步对比格式难以学习，新版累积路径前缀（FrozenLake 风格）让 `<answer>` 直接等于 CoT 最后一行
2. **图像数量一致性**：训练用多张 checkpoint 图、推理只传 base image 会导致训练/推理不一致，应统一只用 base image
3. **grid_state_loss**：weight=0 时 grid_token hidden state 无任何监督，应开启（weight=1.0）
4. **vision tower 冻结**：MiniGrid 图像对比度低，冻结时模型无法从图读出坐标，产生幻觉；新图像风格（橙色障碍）有助于缓解
5. **过拟合**：2000 samples × 4000 steps ≈ 8 epoch，loss 在 step ~500 即收敛，后续死记硬背
