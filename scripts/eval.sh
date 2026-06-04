#!/bin/bash
# Evaluate model on grid CoT dataset

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR/../src:$PYTHONPATH"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH}"
DATASET_PATH="${DATASET_PATH:-dataset/grid_cot_dataset.json}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results/$(date +%Y-%m-%d)}"
SAMPLE_INDICES="${SAMPLE_INDICES:-0 1 2 3 4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
USE_TEXTUAL_STATE_DESC="${USE_TEXTUAL_STATE_DESC:-False}"

python -c "
import argparse, json, os, time, sys, torch
from datetime import datetime, timezone
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

sys.path.insert(0, '$SCRIPT_DIR/../src')
from model import GridCoTForConditionalGeneration
from constants import GRID_SYSTEM_MESSAGE, build_grid_user_prompt, infer_grid_size_from_text
from inference import clean_output, extract_answer, build_suppress_token_ids, SuppressPadTokens

# Load model
processor = AutoProcessor.from_pretrained('$MODEL_PATH')
model = GridCoTForConditionalGeneration.from_pretrained(
    '$MODEL_PATH', torch_dtype=torch.float16, device_map='cuda',
    attn_implementation='flash_attention_2',
)
model.eval()

suppress_ids = build_suppress_token_ids(processor.tokenizer)

with open('$DATASET_PATH') as f:
    dataset = json.load(f)

os.makedirs('$OUTPUT_DIR', exist_ok=True)
records = []

for idx in [${SAMPLE_INDICES// /, }]:
    sample = dataset[idx]
    conversations = sample['conversations']
    reference = conversations[-1]['value'] if len(conversations) >= 2 else ''
    rows, cols = infer_grid_size_from_text(reference)
    user_text = build_grid_user_prompt(
        rows=rows,
        cols=cols,
    )
    
    image_path = sample['images'][0]
    messages = [
        {'role': 'system', 'content': [{'type': 'text', 'text': GRID_SYSTEM_MESSAGE}]},
        {'role': 'user', 'content': [
            {'type': 'image', 'image': image_path},
            {'type': 'text', 'text': user_text},
        ]},
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[prompt], images=image_inputs, padding=True, return_tensors='pt').to('cuda')

    start = time.time()
    with torch.inference_mode():
        gen_ids = model.generate(
            **inputs, do_sample=False, max_new_tokens=$MAX_NEW_TOKENS,
            logits_processor=[SuppressPadTokens(suppress_ids)] if suppress_ids else None,
        )
    elapsed = time.time() - start

    trimmed = gen_ids[0][inputs.input_ids.shape[1]:]
    raw = processor.decode(trimmed, skip_special_tokens=False)
    clean = processor.decode(trimmed, skip_special_tokens=True)
    answer = extract_answer(raw)
    
    records.append({
        'sample_index': idx,
        'inference_seconds': round(elapsed, 2),
        'raw_output': raw,
        'clean_output': clean_output(raw),
        'extracted_answer': answer,
        'reference': reference,
    })
    print(f'Sample {idx}: {answer} ({elapsed:.1f}s)')

with open(os.path.join('$OUTPUT_DIR', 'results.json'), 'w') as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
print(f'Results saved to $OUTPUT_DIR/results.json')
"
