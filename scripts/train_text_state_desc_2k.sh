#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate covt

cd "$PROJECT_DIR"

export MASTER_PORT="${MASTER_PORT:-22810}"
export GPU_IDS="${GPU_IDS:-0}"
export BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"

export BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct}"
export DATA_PATH="${DATA_PATH:-/root/CoVT/visual-cot/dataset/grid_cot_state_only_2k.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/visual-cot_output/grid_cot_text_state_desc_2k_8k}"

export MAX_STEPS="${MAX_STEPS:-8000}"
export SAVE_STEPS="${SAVE_STEPS:-4000}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-True}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero2.json}"

export IMAGE_RESIZED_WIDTH="${IMAGE_RESIZED_WIDTH:-560}"
export IMAGE_RESIZED_HEIGHT="${IMAGE_RESIZED_HEIGHT:-560}"
export IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-250880}"
export IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-1003520}"

export USE_GRID_TOKENS="${USE_GRID_TOKENS:-False}"
export USE_TEXTUAL_STATE_DESC="${USE_TEXTUAL_STATE_DESC:-True}"
export GRID_VISIBLE_TOKEN_LOSS_WEIGHT="${GRID_VISIBLE_TOKEN_LOSS_WEIGHT:-1.0}"
export GRID_PAD_TOKEN_LOSS_WEIGHT="${GRID_PAD_TOKEN_LOSS_WEIGHT:-1.0}"
export ANSWER_TOKEN_LOSS_WEIGHT="${ANSWER_TOKEN_LOSS_WEIGHT:-5.0}"
export ANSWER_SPAN_LOSS_WEIGHT="${ANSWER_SPAN_LOSS_WEIGHT:-2.0}"
export ANSWER_TRANSITION_LOSS_WEIGHT="${ANSWER_TRANSITION_LOSS_WEIGHT:-8.0}"
export GRID_STATE_LOSS_WEIGHT="${GRID_STATE_LOSS_WEIGHT:-0.0}"

echo "=== Train Config ==="
echo "BASE_MODEL=$BASE_MODEL"
echo "DATA_PATH=$DATA_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "GPU_IDS=$GPU_IDS"
echo "BATCH_PER_DEVICE=$BATCH_PER_DEVICE"
echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
echo "MAX_STEPS=$MAX_STEPS"
echo "SAVE_STEPS=$SAVE_STEPS"
echo "USE_TEXTUAL_STATE_DESC=$USE_TEXTUAL_STATE_DESC"
echo "GRID_STATE_LOSS_WEIGHT=$GRID_STATE_LOSS_WEIGHT"
echo

bash scripts/train.sh

echo
echo "==== Training finished ===="
echo "To merge LoRA, run:"
echo "MODEL_PATH=$OUTPUT_DIR MODEL_BASE=$BASE_MODEL SAVE_MODEL_PATH=${OUTPUT_DIR}_merged bash scripts/merge_lora.sh"
