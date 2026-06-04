#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate covt

cd "$PROJECT_DIR"

export MASTER_PORT="${MASTER_PORT:-22840}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

NUM_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
export BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
PER_STEP_BATCH=$((BATCH_PER_DEVICE * NUM_DEVICES))
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / PER_STEP_BATCH))

export BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/visual-cot_output/grid_cot_state_only_500_bfs_bs4_s1000_merged}"
export DATA_PATH="${DATA_PATH:-/root/CoVT/visual-cot/dataset/grid_cot_state_only_2k_bfs_rand500_pref.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/visual-cot_output/grid_cot_state_only_500_bfs_preference_from_best}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero2.json}"

export MAX_STEPS="${MAX_STEPS:-500}"
export SAVE_STEPS="${SAVE_STEPS:-100000}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-True}"
export LEARNING_RATE="${LEARNING_RATE:-2e-6}"
export PROJECTION_LAYER_LR="${PROJECTION_LAYER_LR:-1e-6}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
export PREFERENCE_BETA="${PREFERENCE_BETA:-0.1}"

deepspeed \
    --master_port "$MASTER_PORT" \
    src/train_preference.py \
    --use_liger True \
    --lora_enable True \
    --vision_lora True \
    --lora_namespan_exclude "['visual']" \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --model_id "$BASE_MODEL" \
    --model_path "$BASE_MODEL" \
    --data_path "$DATA_PATH" \
    --freeze_vision_tower True \
    --freeze_llm True \
    --tune_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size "$BATCH_PER_DEVICE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --image_min_pixels 250880 \
    --image_max_pixels 1003520 \
    --image_resized_width 560 \
    --image_resized_height 560 \
    --learning_rate "$LEARNING_RATE" \
    --projection_layer_lr "$PROJECTION_LAYER_LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing True \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 1 \
    --save_only_model "$SAVE_ONLY_MODEL" \
    --dataloader_num_workers 0 \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --report_to none \
    --run_name "grid_cot_preference_train" \
    --use_grid_tokens True \
    --use_textual_state_desc False \
    --preference_beta "$PREFERENCE_BETA"
