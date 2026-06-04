#!/bin/bash
# Single-stage training for GridCoT (pure text CoT, no anchor models)

export MASTER_PORT="${MASTER_PORT:-22810}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src:$PYTHONPATH"

NUM_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
PER_STEP_BATCH=$((BATCH_PER_DEVICE * NUM_DEVICES))
if (( PER_STEP_BATCH <= 0 )); then
    echo "[error] Invalid batch setup: BATCH_PER_DEVICE=$BATCH_PER_DEVICE, NUM_DEVICES=$NUM_DEVICES" >&2
    exit 1
fi
if (( GLOBAL_BATCH_SIZE < PER_STEP_BATCH )); then
    echo "[error] GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE is smaller than per-step batch=$PER_STEP_BATCH" >&2
    exit 1
fi
if (( GLOBAL_BATCH_SIZE % PER_STEP_BATCH != 0 )); then
    echo "[error] GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE is not divisible by per-step batch=$PER_STEP_BATCH" >&2
    exit 1
fi
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / PER_STEP_BATCH))

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/visual-cot_output/grid_cot}"
DATA_PATH="${DATA_PATH:-dataset/grid_cot_dataset.json}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero2.json}"
MAX_STEPS="${MAX_STEPS:-4000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-False}"

# Loss weights (key improvements over old project)
GRID_VISIBLE_TOKEN_LOSS_WEIGHT="${GRID_VISIBLE_TOKEN_LOSS_WEIGHT:-1.0}"
GRID_PAD_TOKEN_LOSS_WEIGHT="${GRID_PAD_TOKEN_LOSS_WEIGHT:-1.0}"
ANSWER_TOKEN_LOSS_WEIGHT="${ANSWER_TOKEN_LOSS_WEIGHT:-5.0}"
ANSWER_SPAN_LOSS_WEIGHT="${ANSWER_SPAN_LOSS_WEIGHT:-2.0}"
ANSWER_TRANSITION_LOSS_WEIGHT="${ANSWER_TRANSITION_LOSS_WEIGHT:-8.0}"
GRID_STATE_LOSS_WEIGHT="${GRID_STATE_LOSS_WEIGHT:-0.5}"
USE_GRID_TOKENS="${USE_GRID_TOKENS:-True}"
USE_TEXTUAL_STATE_DESC="${USE_TEXTUAL_STATE_DESC:-False}"

# Optimization (overridable to stabilize training)
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
PROJECTION_LAYER_LR="${PROJECTION_LAYER_LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# Image settings
IMAGE_RESIZED_WIDTH="${IMAGE_RESIZED_WIDTH:-560}"
IMAGE_RESIZED_HEIGHT="${IMAGE_RESIZED_HEIGHT:-560}"
IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-250880}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-1003520}"

deepspeed \
    --master_port "$MASTER_PORT" \
    src/train.py \
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
    --image_min_pixels "$IMAGE_MIN_PIXELS" \
    --image_max_pixels "$IMAGE_MAX_PIXELS" \
    --image_resized_width "$IMAGE_RESIZED_WIDTH" \
    --image_resized_height "$IMAGE_RESIZED_HEIGHT" \
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
    --run_name "grid_cot_train" \
    --use_grid_tokens "$USE_GRID_TOKENS" \
    --use_textual_state_desc "$USE_TEXTUAL_STATE_DESC" \
    --grid_visible_token_loss_weight "$GRID_VISIBLE_TOKEN_LOSS_WEIGHT" \
    --grid_pad_token_loss_weight "$GRID_PAD_TOKEN_LOSS_WEIGHT" \
    --answer_token_loss_weight "$ANSWER_TOKEN_LOSS_WEIGHT" \
    --answer_span_loss_weight "$ANSWER_SPAN_LOSS_WEIGHT" \
    --answer_transition_loss_weight "$ANSWER_TRANSITION_LOSS_WEIGHT" \
    --grid_state_loss_weight "$GRID_STATE_LOSS_WEIGHT"
