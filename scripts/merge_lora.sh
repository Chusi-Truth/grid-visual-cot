#!/bin/bash
# Merge LoRA weights into base model

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR/../src:$PYTHONPATH"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to LoRA checkpoint dir}"
MODEL_BASE="${MODEL_BASE:?Set MODEL_BASE to base model path}"
SAVE_MODEL_PATH="${SAVE_MODEL_PATH:?Set SAVE_MODEL_PATH to output path}"

python src/merge_lora.py \
    --model-path "$MODEL_PATH" \
    --model-base "$MODEL_BASE" \
    --save-model-path "$SAVE_MODEL_PATH" \
    --safe-serialization
