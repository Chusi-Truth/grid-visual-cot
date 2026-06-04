"""Merge LoRA weights into base model."""

import argparse
import os
import torch
from peft import PeftModel
from transformers import AutoProcessor

from model import GridCoTForConditionalGeneration
from constants import NEW_TOKENS
from trainer import TRAINABLE_TOKEN_ROWS_NAME, load_trainable_token_rows


def _has_adapter_weights(model_path):
    return any(
        os.path.exists(os.path.join(model_path, name))
        for name in ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]
    )


def merge_lora(model_path, model_base, save_path, safe_serialization=True):
    # Load processor from checkpoint (has new tokens)
    try:
        processor = AutoProcessor.from_pretrained(model_path)
    except Exception:
        processor = AutoProcessor.from_pretrained(model_base)
        processor.tokenizer.add_tokens(NEW_TOKENS)

    # Load base model
    model = GridCoTForConditionalGeneration.from_pretrained(
        model_base,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    # Resize embeddings to match tokenizer
    target_size = len(processor.tokenizer)
    current_size = model.get_input_embeddings().weight.shape[0]
    if current_size < target_size:
        model.resize_token_embeddings(target_size)

    # Load LoRA adapter
    if _has_adapter_weights(model_path):
        print(f"Loading LoRA adapter from {model_path}")
        model = PeftModel.from_pretrained(model, model_path, device_map="cpu")

        # Load non-LoRA state dict (new token embeddings, etc.)
        non_lora_path = os.path.join(model_path, "non_lora_state_dict.bin")
        if os.path.exists(non_lora_path):
            non_lora_state = torch.load(non_lora_path, map_location="cpu")
            model.load_state_dict(non_lora_state, strict=False)
            print(f"Loaded non-LoRA weights: {list(non_lora_state.keys())[:5]}...")
        elif os.path.exists(os.path.join(model_path, TRAINABLE_TOKEN_ROWS_NAME)):
            load_trainable_token_rows(model, model_path)
            print(f"Loaded trainable token rows from {TRAINABLE_TOKEN_ROWS_NAME}")

        model = model.merge_and_unload()
        print("LoRA merged successfully")
    else:
        print(f"No adapter found at {model_path}, loading as full model")
        model = GridCoTForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
        )

    model.save_pretrained(save_path, safe_serialization=safe_serialization)
    processor.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--model-base", required=True, help="Path to base model")
    parser.add_argument("--save-model-path", required=True, help="Output path")
    parser.add_argument("--safe-serialization", action="store_true")
    args = parser.parse_args()

    merge_lora(args.model_path, args.model_base, args.save_model_path, args.safe_serialization)
