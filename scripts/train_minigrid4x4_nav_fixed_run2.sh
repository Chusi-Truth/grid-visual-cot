#!/bin/bash
# run2: 修复 detect_env_type 误判问题（minigrid_nav 被错判为 minigrid）
# 现在训练时的系统消息与 eval 完全一致

set -euo pipefail

cd /root/CoVT/visual-cot

source /root/miniconda3/etc/profile.d/conda.sh
conda activate covt

deepspeed --include localhost:0 src/train.py \
  --use_liger True \
  --lora_enable True \
  --vision_lora True \
  --lora_namespan_exclude "['visual']" \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --model_id /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct \
  --model_path /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct \
  --data_path /root/CoVT/visual-cot/dataset/minigrid4x4_nav_fixed_1000.json \
  --freeze_vision_tower True \
  --freeze_llm True \
  --tune_merger False \
  --bf16 True \
  --fp16 False \
  --disable_flash_attn2 False \
  --output_dir /root/autodl-tmp/visual-cot_output/minigrid4x4_nav_fixed_1000_run2 \
  --max_steps 2000 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --image_min_pixels 250880 \
  --image_max_pixels 1003520 \
  --image_resized_width 560 \
  --image_resized_height 560 \
  --learning_rate 5e-6 \
  --projection_layer_lr 1e-6 \
  --weight_decay 0.05 \
  --max_grad_norm 0.3 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --tf32 True \
  --gradient_checkpointing True \
  --save_strategy no \
  --save_total_limit 1 \
  --save_only_model True \
  --dataloader_num_workers 0 \
  --deepspeed scripts/zero2.json \
  --report_to none \
  --run_name minigrid4x4_nav_fixed_1000_run2 \
  --use_grid_tokens True \
  --use_textual_state_desc False \
  --grid_visible_token_loss_weight 1.0 \
  --grid_pad_token_loss_weight 1.0 \
  --answer_token_loss_weight 10.0 \
  --answer_span_loss_weight 20.0 \
  --answer_transition_loss_weight 15.0 \
  --grid_state_loss_weight 1.0

# ── Save hyperparameters ──────────────────────────────────────────────────
OUTPUT_DIR=/root/autodl-tmp/visual-cot_output/minigrid4x4_nav_fixed_1000_run2
cat > "${OUTPUT_DIR}/hparams.md" << 'EOF'
# Hyperparameters

| 参数 | 值 |
|---|---|
| run_name | minigrid4x4_nav_fixed_1000_run2 |
| dataset | minigrid4x4_nav_fixed_1000.json (1000 samples) |
| 实验目的 | 修复 detect_env_type 误判 bug 后重新训练（run1 用了 DoorKey 系统消息） |
| 关键修复 | data.py: 优先读 meta.env_type，避免 CoT 中的 turn left/go forward 误判为 minigrid |
| CoT格式 | v2: FrozenLake风格累积路径，base_image only |
| 图像风格 | PIL自绘: 深灰背景+橙色波浪障碍+亮绿goal+红色箭头 |
| use_grid_tokens | True |
| freeze_vision_tower | True |
| freeze_llm | True |
| lora_rank / alpha | 16 / 32 |
| lora_dropout | 0.05 |
| max_steps | 2000 |
| per_device_train_batch_size | 4 |
| learning_rate | 5e-6 |
| weight_decay | 0.05 |
| answer_span_loss_weight | 20.0 |
| answer_transition_loss_weight | 15.0 |
| grid_state_loss_weight | 1.0 |
EOF
echo "hparams saved to ${OUTPUT_DIR}/hparams.md"
