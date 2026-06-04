#!/bin/bash
# text-only: 不用grid_token，纯文本CoT，和run4(state_only)同样训练量对比

set -euo pipefail
cd /root/CoVT/visual-cot
source /root/miniconda3/etc/profile.d/conda.sh
conda activate covt

OUTPUT_DIR=/root/autodl-tmp/visual-cot_output/minigrid4x4_nav_v2_2000_textonly_run1
mkdir -p "${OUTPUT_DIR}"

deepspeed --include localhost:0 src/train.py \
  --use_liger True \
  --lora_enable True \
  --vision_lora True \
  --lora_namespan_exclude "['visual']" \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.01 \
  --model_id /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct \
  --model_path /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct \
  --data_path /root/CoVT/visual-cot/dataset/minigrid4x4_nav_v2_2000_baseonly.json \
  --freeze_vision_tower True \
  --freeze_llm True \
  --tune_merger False \
  --bf16 True \
  --fp16 False \
  --disable_flash_attn2 False \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps 8000 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --image_min_pixels 250880 \
  --image_max_pixels 1003520 \
  --image_resized_width 560 \
  --image_resized_height 560 \
  --learning_rate 5e-5 \
  --projection_layer_lr 1e-5 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
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
  --run_name minigrid4x4_nav_v2_2000_textonly_run1 \
  --use_grid_tokens False \
  --use_textual_state_desc False \
  --grid_visible_token_loss_weight 1.0 \
  --grid_pad_token_loss_weight 1.0 \
  --answer_token_loss_weight 5.0 \
  --answer_span_loss_weight 2.0 \
  --answer_transition_loss_weight 8.0 \
  --grid_state_loss_weight 0.0

cat > "${OUTPUT_DIR}/hparams.md" << 'EOF'
# Hyperparameters — minigrid4x4_nav_v2_2000_textonly_run1

| 参数 | 值 | 对比 run4(state_only) |
|---|---|---|
| 实验目的 | text-only消融：不用grid_token，纯文本CoT | - |
| dataset | minigrid4x4_nav_v2_2000_baseonly.json (2000 samples) | 同 |
| max_steps | 8000 | 同 |
| learning_rate | 5e-5 | 同 |
| use_grid_tokens | **False** | run4: True |
| grid_state_loss_weight | **0.0** | run4: 0.5 |
| 其余超参 | 全部同 run4 | - |
EOF
echo "hparams saved to ${OUTPUT_DIR}/hparams.md"
