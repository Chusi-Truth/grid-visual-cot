# Visual-CoT: Grid Planning with Visual Chain-of-Thought

基于 Qwen2.5-VL 的 FrozenLake 网格路径规划训练框架。通过加权 CE loss 训练模型生成结构化的 `<think>/<answer>` CoT 推理格式。

## 项目结构

```
visual-cot/
├── scripts/
│   ├── build_dataset.py      # 数据集构建（beam search + checkpoint 渲染）
│   ├── train.sh              # 训练启动脚本
│   ├── merge_lora.sh         # LoRA 合并
│   ├── eval.sh               # 评估脚本
│   ├── zero2.json            # DeepSpeed ZeRO-2 配置
│   └── zero3.json            # DeepSpeed ZeRO-3 配置
├── src/
│   ├── constants.py          # Token 定义、prompt 模板
│   ├── model.py              # GridCoT 模型（加权 CE loss）
│   ├── data.py               # 数据加载、grid token 处理
│   ├── train.py              # 训练入口
│   ├── trainer.py            # 自定义 Trainer
│   ├── inference.py          # 推理、constrained decoding
│   └── merge_lora.py         # LoRA 合并脚本
├── dataset/                  # 数据集存放目录
└── requirements.txt
```

## 环境安装

```bash
pip install -r requirements.txt
```

需要的核心依赖：`transformers>=4.45.0`, `peft>=0.7.0`, `deepspeed>=0.14.0`, `qwen-vl-utils`

## 快速开始

### 1. 构建数据集

```bash
python scripts/build_dataset.py \
    --sample_num 500 \
    --checkpoint_interval 2 \
    --output_json dataset/grid_cot_dataset.json \
    --checkpoint_dir dataset/grid_checkpoints
```

生成的 JSON 格式：
```json
{
  "conversations": [
    {"from": "human", "value": "<image>"},
    {"from": "gpt", "value": "<think>...推理过程...<grid_token>...继续推理...</think>\n<answer>go right -> go down -> ...</answer>"}
  ],
  "images": ["base_grid.jpg", "checkpoint_1.png", "checkpoint_2.png"]
}
```

- `images[0]`：基础网格图（输入给 VLM）
- `images[1:]`：推理过程中的 checkpoint 快照（当前仅保留用于未来扩展）

### 2. 训练

**单卡训练：**

```bash
cd visual-cot

BASE_MODEL=/path/to/Qwen2.5-VL-7B-Instruct \
DATA_PATH=dataset/grid_cot_dataset.json \
OUTPUT_DIR=output/grid_cot \
bash scripts/train.sh
```

**多卡训练：**

```bash
GPU_IDS=0,1,2,3 \
BATCH_PER_DEVICE=2 \
GLOBAL_BATCH_SIZE=8 \
BASE_MODEL=/path/to/Qwen2.5-VL-7B-Instruct \
DATA_PATH=dataset/grid_cot_dataset.json \
OUTPUT_DIR=output/grid_cot \
bash scripts/train.sh
```

**可调参数（通过环境变量）：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BASE_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` | 基础模型路径 |
| `DATA_PATH` | `dataset/grid_cot_dataset.json` | 训练数据路径 |
| `OUTPUT_DIR` | `output/grid_cot` | 输出目录 |
| `MAX_STEPS` | `4000` | 总训练步数 |
| `SAVE_STEPS` | `2000` | 保存 checkpoint 间隔 |
| `GPU_IDS` | `0` | 使用的 GPU |
| `BATCH_PER_DEVICE` | `2` | 每卡 batch size |
| `GLOBAL_BATCH_SIZE` | `2` | 全局 batch size |
| `DEEPSPEED_CONFIG` | `scripts/zero2.json` | DeepSpeed 配置 |
| `GRID_VISIBLE_TOKEN_LOSS_WEIGHT` | `4.0` | `<grid_token>` 的 loss 权重 |
| `ANSWER_TOKEN_LOSS_WEIGHT` | `5.0` | `<answer>`/`</answer>` 标签的 loss 权重 |
| `ANSWER_SPAN_LOSS_WEIGHT` | `2.0` | answer 内容区间的 loss 权重 |
| `ANSWER_TRANSITION_LOSS_WEIGHT` | `8.0` | `</think>` → `<answer>` 转换的 loss 权重 |

### 3. 合并 LoRA

训练完成后，将 LoRA 权重合并到基础模型中：

```bash
MODEL_PATH=output/grid_cot \
MODEL_BASE=/path/to/Qwen2.5-VL-7B-Instruct \
SAVE_MODEL_PATH=output/grid_cot_merged \
bash scripts/merge_lora.sh
```

### 4. 评估

```bash
MODEL_PATH=output/grid_cot_merged \
DATASET_PATH=dataset/grid_cot_dataset.json \
OUTPUT_DIR=eval_results/$(date +%Y-%m-%d) \
SAMPLE_INDICES="0 1 2 3 4" \
bash scripts/eval.sh
```

评估结果保存在 `OUTPUT_DIR/results.json`。

## 核心设计

### Special Tokens

本项目只使用 6 个 special token，避免了旧项目中 14 个 token 导致的推理泄漏问题：

| Token | 用途 |
|-------|------|
| `<grid_token>` | 视觉 checkpoint 可见标记，模型主动生成 |
| `<\|grid_pad\|>` | 隐式填充（每个 checkpoint 展开为 1 visible + 7 pad = 8 token） |
| `<think>` | 推理开始 |
| `</think>` | 推理结束 |
| `<answer>` | 答案开始 |
| `</answer>` | 答案结束 |

### 加权 CE Loss

对不同类型的 token 施加不同的 loss 权重，引导模型学习正确的输出结构：

- **`</think>` → `<answer>` 转换**（权重 8.0）：最关键的格式转折点
- **`<answer>` / `</answer>` 标签**（权重 5.0）：确保答案区间完整
- **`<grid_token>`**（权重 4.0）：引导模型在正确位置生成 checkpoint
- **answer 内容区间**（权重 2.0）：提升路径输出质量

### Constrained Decoding

推理时提供两种约束机制：

1. **SuppressPadTokens**：禁止生成 `<|grid_pad|>` 等 pad token，防止泄漏
2. **GridFormatEnforcer**（可选）：状态机强制输出遵循 `<think>→</think>→<answer>→</answer>` 格式

### 模型输出示例

```
<think>
The map size is 4 rows by 5 columns. Starting at (0,0), need to reach goal at (3,4).
<grid_token>
Analyzing paths: going right then down avoids the hole at (1,1).
<grid_token>
From (2,3), the safest route is right then down to reach the goal.
</think>
<answer>go right -> go right -> go down -> go right -> go right -> go down -> go down</answer>
```

## 与旧项目 (train/) 的区别

| 对比项 | 旧项目 (train/) | 新项目 (visual-cot/) |
|--------|-----------------|---------------------|
| Special tokens | 14 个（含 8 个 anchor pad） | 6 个 |
| Anchor models | SigLIP + 8 个 anchor | 无 |
| Loss | CE + MSE alignment | 纯 CE（加权） |
| 训练阶段 | 2 阶段 + 中间 merge | 1 阶段 |
| 代码量 | ~3000 行 | ~1400 行 |
| Token 泄漏风险 | 高 | 低 |
