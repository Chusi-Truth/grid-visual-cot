#!/bin/bash
# DoorKey 4x4 完整训练，state_only 和 text_only 对比

set -euo pipefail
cd /root/CoVT/visual-cot
source /root/miniconda3/etc/profile.d/conda.sh
conda activate covt

MODE="${1:-state_only}"

if [ "$MODE" = "text_only" ]; then
  USE_GRID=False
  GRID_LOSS=0.0
  SUFFIX=textonly
else
  USE_GRID=True
  GRID_LOSS=0.5
  SUFFIX=stateonly
fi

OUTPUT_DIR=/root/autodl-tmp/visual-cot_output/minigrid4x4_doorkey_2000_${SUFFIX}
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
  --data_path /root/CoVT/visual-cot/dataset/minigrid4x4_doorkey_2000.json \
  --freeze_vision_tower True \
  --freeze_llm True \
  --tune_merger False \
  --bf16 True \
  --fp16 False \
  --disable_flash_attn2 False \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps 4000 \
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
  --run_name minigrid4x4_doorkey_2000_${SUFFIX} \
  --use_grid_tokens ${USE_GRID} \
  --use_textual_state_desc False \
  --grid_visible_token_loss_weight 1.0 \
  --grid_pad_token_loss_weight 1.0 \
  --answer_token_loss_weight 5.0 \
  --answer_span_loss_weight 2.0 \
  --answer_transition_loss_weight 8.0 \
  --grid_state_loss_weight ${GRID_LOSS}

cat > "${OUTPUT_DIR}/hparams.md" << EOF
# Hyperparameters — minigrid4x4_doorkey_2000_${SUFFIX}

| 参数 | 值 |
|---|---|
| 实验目的 | DoorKey 4x4 完整训练 (${MODE}) |
| dataset | minigrid4x4_doorkey_2000.json (2000 samples) |
| max_steps | 8000 |
| learning_rate | 5e-5 |
| use_grid_tokens | ${USE_GRID} |
| grid_state_loss_weight | ${GRID_LOSS} |
EOF
echo "hparams saved to ${OUTPUT_DIR}/hparams.md"
