"""Pairwise preference training for GridCoT.

This keeps the existing model architecture and LoRA setup, but replaces the
standard SFT loss with a simple pairwise ranking loss between chosen/rejected
assistant responses.
"""

import ast
import copy
import os
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional

from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor, HfArgumentParser, TrainingArguments as HfTrainingArguments

from deepspeed import zero

from constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    NEW_TOKENS,
    GRID_SYSTEM_MESSAGE,
    GRID_SYSTEM_MESSAGE_TEXT_ONLY,
    GRID_SYSTEM_MESSAGE_STATE_DESC,
    infer_grid_size_from_text,
    build_grid_user_prompt,
)
from data import llava_to_openai, normalize_reasoning_tags, pad_sequence
from model import GridCoTForConditionalGeneration
from trainer import GridCoTTrainer, get_peft_state_non_lora_maybe_zero_3


local_rank = None
torch.manual_seed(42)


def rank0_print(*args):
    if local_rank in (0, "0", None):
        print(*args)


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=None):
    lora_namespan_exclude = lora_namespan_exclude or []
    names = []
    for name, module in model.named_modules():
        if any(ex in name for ex in lora_namespan_exclude):
            continue
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            names.append(name)
    if num_lora_modules > 0:
        names = names[-num_lora_modules:]
    return names


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


@dataclass
class ModelArguments:
    model_id: str = field(default="Qwen/Qwen2.5-VL-7B-Instruct")
    model_path: str = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    image_min_pixels: Optional[int] = field(default=3136)
    image_max_pixels: Optional[int] = field(default=12845056)
    image_resized_width: int = field(default=None)
    image_resized_height: int = field(default=None)
    use_grid_tokens: bool = field(default=True)
    use_textual_state_desc: bool = field(default=False)


@dataclass
class TrainingArguments(HfTrainingArguments):
    freeze_vision_tower: bool = False
    freeze_llm: bool = False
    tune_merger: bool = False
    disable_flash_attn2: bool = False
    max_seq_length: int = 32768

    lora_enable: bool = False
    vision_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.01
    lora_bias: str = "none"
    lora_namespan_exclude: str = field(default=None)
    num_lora_modules: int = -1
    use_liger: bool = True

    vision_lr: Optional[float] = None
    projection_layer_lr: Optional[float] = None

    preference_beta: float = 0.1
    preference_weight_scale: float = 1.0


class PreferenceGridCoTDataset(Dataset):
    def __init__(self, data_path, processor, data_args):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = __import__("json").load(f)
        self.processor = processor
        self.data_args = data_args
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.use_grid_tokens = data_args.use_grid_tokens
        self.use_textual_state_desc = data_args.use_textual_state_desc

    def __len__(self):
        return len(self.data)

    def _default_system(self):
        if self.use_textual_state_desc:
            return GRID_SYSTEM_MESSAGE_STATE_DESC
        if self.use_grid_tokens:
            return GRID_SYSTEM_MESSAGE
        return GRID_SYSTEM_MESSAGE_TEXT_ONLY

    def _load_image(self, sample):
        all_paths = sample.get("images") or []
        if isinstance(all_paths, str):
            all_paths = [all_paths]
        if all_paths:
            return Image.open(all_paths[0]).convert("RGB")
        return Image.new("RGB", (self.image_resized_w or 560, self.image_resized_h or 560), (0, 0, 0))

    def _build_prompt(self, sample):
        sources = copy.deepcopy(llava_to_openai(sample["conversations"]))
        system_content = self._default_system()
        if sources and sources[0]["role"] == "system":
            system_content = sources[0]["content"] or system_content
            sources = sources[1:]
        if len(sources) < 2:
            raise ValueError("Expected at least one user/assistant turn")

        user_input = sources[0]
        reference_response = sources[1]["content"]
        rows, cols = infer_grid_size_from_text(reference_response)
        prompt_text = build_grid_user_prompt(
            rows=rows,
            cols=cols,
            include_grid_token=self.use_grid_tokens and not self.use_textual_state_desc,
            use_textual_state_desc=self.use_textual_state_desc,
        )
        user_content = user_input["content"]
        if DEFAULT_IMAGE_TOKEN in user_content:
            user_content = f"{user_content}\n{prompt_text}"
        else:
            user_content = f"{user_content.strip()}\n{prompt_text}" if user_content.strip() else prompt_text

        prompt = (
            f"{DEFAULT_IM_START_TOKEN}system\n{system_content}\n{DEFAULT_IM_END_TOKEN}\n"
            f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_content}\n{DEFAULT_IM_END_TOKEN}\n"
            f"{DEFAULT_IM_START_TOKEN}assistant\n"
        )
        return prompt

    def _tokenize_pair(self, prompt, response_text, image):
        response_text = normalize_reasoning_tags(response_text)
        response_str = f"{response_text}\n{DEFAULT_IM_END_TOKEN}\n"
        inputs = self.processor(text=[prompt], images=[image], padding=False, return_tensors="pt")
        prompt_ids = inputs["input_ids"]
        response_ids = self.processor.tokenizer(
            response_str, add_special_tokens=False, padding=False, return_tensors="pt"
        )["input_ids"]
        input_ids = torch.cat([prompt_ids, response_ids], dim=1).squeeze(0)
        labels = torch.cat(
            [
                torch.full((prompt_ids.shape[1],), IGNORE_INDEX, dtype=torch.long),
                response_ids.squeeze(0),
            ]
        )
        return {
            "input_ids": input_ids.to(torch.long),
            "labels": labels.to(torch.long),
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
        }

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = self._load_image(sample)
        prompt = self._build_prompt(sample)
        chosen = self._tokenize_pair(prompt, sample["chosen_response"], image)
        rejected = self._tokenize_pair(prompt, sample["rejected_response"], image)
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_labels": chosen["labels"],
            "chosen_pixel_values": chosen["pixel_values"],
            "chosen_image_grid_thw": chosen["image_grid_thw"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_labels": rejected["labels"],
            "rejected_pixel_values": rejected["pixel_values"],
            "rejected_image_grid_thw": rejected["image_grid_thw"],
            "pair_weight": float(sample.get("chosen_reward", 1.0) - sample.get("rejected_reward", 0.0)),
        }


class PreferenceDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        merged_input_ids = pad_sequence(
            [ex["chosen_input_ids"] for ex in examples] + [ex["rejected_input_ids"] for ex in examples],
            padding_side="right",
            padding_value=self.pad_token_id,
        )
        merged_labels = pad_sequence(
            [ex["chosen_labels"] for ex in examples] + [ex["rejected_labels"] for ex in examples],
            padding_side="right",
            padding_value=IGNORE_INDEX,
        )
        batch_size = len(examples)
        chosen_input_ids = merged_input_ids[:batch_size]
        rejected_input_ids = merged_input_ids[batch_size:]
        chosen_labels = merged_labels[:batch_size]
        rejected_labels = merged_labels[batch_size:]
        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_input_ids.ne(self.pad_token_id),
            "chosen_labels": chosen_labels,
            "chosen_pixel_values": torch.cat([ex["chosen_pixel_values"] for ex in examples], dim=0),
            "chosen_image_grid_thw": torch.cat([ex["chosen_image_grid_thw"] for ex in examples], dim=0),
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_input_ids.ne(self.pad_token_id),
            "rejected_labels": rejected_labels,
            "rejected_pixel_values": torch.cat([ex["rejected_pixel_values"] for ex in examples], dim=0),
            "rejected_image_grid_thw": torch.cat([ex["rejected_image_grid_thw"] for ex in examples], dim=0),
            "pair_weight": torch.tensor([ex["pair_weight"] for ex in examples], dtype=torch.float32),
        }


class PreferenceTrainer(GridCoTTrainer):
    def __init__(self, *args, preference_beta=0.1, preference_weight_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.preference_beta = preference_beta
        self.preference_weight_scale = preference_weight_scale

    def _sequence_logprob(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        valid_mask = shift_labels.ne(IGNORE_INDEX)
        safe_labels = shift_labels.masked_fill(~valid_mask, 0)
        token_logprobs = F.log_softmax(shift_logits.float(), dim=-1).gather(
            dim=-1, index=safe_labels.unsqueeze(-1)
        ).squeeze(-1)
        token_logprobs = token_logprobs * valid_mask
        lengths = valid_mask.sum(dim=-1).clamp_min(1)
        return token_logprobs.sum(dim=-1) / lengths

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        pair_weight = inputs.pop("pair_weight").to(model.device)
        batch_size = inputs["chosen_input_ids"].shape[0]

        merged_outputs = model(
            input_ids=torch.cat([inputs["chosen_input_ids"], inputs["rejected_input_ids"]], dim=0),
            attention_mask=torch.cat([inputs["chosen_attention_mask"], inputs["rejected_attention_mask"]], dim=0),
            labels=None,
            pixel_values=torch.cat([inputs["chosen_pixel_values"], inputs["rejected_pixel_values"]], dim=0),
            image_grid_thw=torch.cat([inputs["chosen_image_grid_thw"], inputs["rejected_image_grid_thw"]], dim=0),
        )

        chosen_logits = merged_outputs.logits[:batch_size]
        rejected_logits = merged_outputs.logits[batch_size:]

        chosen_logprob = self._sequence_logprob(chosen_logits, inputs["chosen_labels"])
        rejected_logprob = self._sequence_logprob(rejected_logits, inputs["rejected_labels"])
        margins = self.preference_beta * (chosen_logprob - rejected_logprob)
        weights = pair_weight.clamp_min(1e-3) * self.preference_weight_scale
        loss = -(weights * F.logsigmoid(margins)).mean()

        if return_outputs:
            aux = {
                "chosen_logprob": chosen_logprob.detach(),
                "rejected_logprob": rejected_logprob.detach(),
            }
            return loss, aux
        return loss


def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if model_args.model_path is None:
        model_args.model_path = model_args.model_id
    training_args.remove_unused_columns = False

    local_rank = training_args.local_rank
    compute_dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32)

    if training_args.use_liger:
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
            apply_liger_kernel_to_qwen2_vl()
        except ImportError:
            rank0_print("[warn] liger_kernel not available, skipping")

    model = GridCoTForConditionalGeneration.from_pretrained(
        model_args.model_path,
        torch_dtype=compute_dtype,
        attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
    )
    model.config.use_cache = False

    set_requires_grad(model.model.parameters(), not training_args.freeze_llm)
    set_requires_grad(model.lm_head.parameters(), not training_args.freeze_llm)
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=training_args.device)
    set_requires_grad(vision_tower.parameters(), not training_args.freeze_vision_tower)
    set_requires_grad(vision_tower.merger.parameters(), training_args.tune_merger)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    if training_args.lora_enable:
        if not training_args.freeze_llm:
            raise ValueError("lora_enable=True requires freeze_llm=True")
        exclude = ast.literal_eval(training_args.lora_namespan_exclude) if training_args.lora_namespan_exclude else []
        if not training_args.vision_lora:
            exclude.append("visual")
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, training_args.num_lora_modules, exclude),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
        )
        if training_args.bf16:
            model.to(torch.bfloat16)
        rank0_print("Adding LoRA...")
        model = get_peft_model(model, peft_config)

    processor = AutoProcessor.from_pretrained(model_args.model_path, padding_side="right")
    old_vocab_size = len(processor.tokenizer)
    processor.tokenizer.add_tokens(NEW_TOKENS)
    new_vocab_size = len(processor.tokenizer)
    if new_vocab_size > old_vocab_size:
        rank0_print(f"Tokenizer size after adding tokens: {new_vocab_size}")
        embed_weight = model.get_input_embeddings().weight
        if hasattr(embed_weight, "ds_id"):
            with zero.GatheredParameters([embed_weight]):
                current_size = embed_weight.data.shape[0]
        else:
            current_size = embed_weight.data.shape[0]
        if new_vocab_size > current_size:
            model.resize_token_embeddings(new_vocab_size)

    for n, p in model.named_parameters():
        if "embed_tokens" in n or "lm_head" in n:
            p.requires_grad = True

    dataset = PreferenceGridCoTDataset(data_args.data_path, processor, data_args)
    collator = PreferenceDataCollator(pad_token_id=processor.tokenizer.pad_token_id)

    trainer = PreferenceTrainer(
        model=model,
        processor=processor,
        trainable_token_ids=[],
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collator,
        preference_beta=training_args.preference_beta,
        preference_weight_scale=training_args.preference_weight_scale,
    )

    trainer.train()
    trainer.save_state()
    model.config.use_cache = True

    if training_args.lora_enable:
        from train import get_peft_state_maybe_zero_3

        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora = get_peft_state_non_lora_maybe_zero_3(model.named_parameters(), require_grad_only=True)
        if local_rank in (0, -1):
            os.makedirs(training_args.output_dir, exist_ok=True)
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(
                training_args.output_dir,
                state_dict=state_dict,
                safe_serialization=False,
            )
            torch.save(non_lora, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
            rank0_print(f"Saved final preference LoRA to {training_args.output_dir}")
    else:
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        trainer._save(training_args.output_dir, state_dict=state_dict)


if __name__ == "__main__":
    train()
