"""Training entry point for GridCoT (pure text CoT, no anchor models)."""

import os
import ast
import pathlib
import torch
from dataclasses import dataclass, field
from typing import Optional

from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, HfArgumentParser, TrainingArguments as HfTrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from deepspeed import zero

from model import GridCoTForConditionalGeneration
from data import DataArguments, make_data_module
from trainer import (
    GridCoTTrainer,
    ResumeDatasetCallback,
    get_peft_state_non_lora_maybe_zero_3,
)
from constants import (
    NEW_TOKENS, GRID_PAD_TOKEN, GRID_VISIBLE_TOKEN,
    THINK_START, THINK_END, ANSWER_START, ANSWER_END,
)

local_rank = None
torch.manual_seed(42)


def rank0_print(*args):
    if local_rank in (0, "0", None):
        print(*args)


def checkpoint_has_nonfinite_adapter(checkpoint_dir: str) -> bool:
    """Return True if LoRA adapter checkpoint contains NaN/Inf tensors."""
    adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    if not os.path.exists(adapter_path):
        return False

    try:
        from safetensors.torch import load_file
    except Exception as exc:
        rank0_print(f"[warn] cannot import safetensors to validate checkpoint: {exc}")
        return False

    try:
        state = load_file(adapter_path)
        for _, value in state.items():
            if not torch.isfinite(value).all():
                return True
    except Exception as exc:
        rank0_print(f"[warn] failed to validate checkpoint tensors at {adapter_path}: {exc}")
        return False
    return False


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_id: str = field(default="Qwen/Qwen2.5-VL-7B-Instruct")
    model_path: str = field(default=None)


@dataclass
class TrainingArguments(HfTrainingArguments):
    # Freezing
    freeze_vision_tower: bool = False
    freeze_llm: bool = False
    tune_merger: bool = False
    disable_flash_attn2: bool = False
    max_seq_length: int = 32768

    # LoRA
    lora_enable: bool = False
    vision_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.01
    lora_bias: str = "none"
    lora_namespan_exclude: str = field(default=None)
    num_lora_modules: int = -1
    use_liger: bool = True

    # LR overrides
    vision_lr: Optional[float] = None
    projection_layer_lr: Optional[float] = None

    # Loss weights
    grid_visible_token_loss_weight: float = 4.0
    grid_pad_token_loss_weight: float = 1.0
    answer_token_loss_weight: float = 5.0
    answer_span_loss_weight: float = 2.0
    answer_transition_loss_weight: float = 8.0
    grid_state_loss_weight: float = 0.5


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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


def force_unfreeze_grid_alignment_modules(model):
    """Always train grid alignment pathway even when freeze_llm=True + LoRA.

    We need the grid-state encoder front-end (including conv layers) to adapt
    when state representation changes (e.g., 6ch -> 10ch). Otherwise, new
    channels are seen through a frozen random projection and performance can
    collapse.
    """
    train_keywords = (
        "grid_state_encoder",
        "grid_hidden_projector",
    )
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        if any(k in name for k in train_keywords):
            total_params += param.numel()
            param.requires_grad = True
            trainable_params += param.numel()
    rank0_print(
        f"[GridAlign] force-unfrozen params: {trainable_params}/{total_params}"
    )


def get_peft_state_maybe_zero_3(named_params, bias):
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    def maybe_zero_3(param):
        if hasattr(param, "ds_id"):
            with zero.GatheredParameters([param]):
                param = param.data.detach().cpu().clone()
        else:
            param = param.detach().cpu().clone()
        return param

    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    else:
        raise NotImplementedError
    return {k: maybe_zero_3(v) for k, v in to_return.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.model_path is None:
        model_args.model_path = model_args.model_id

    local_rank = training_args.local_rank
    compute_dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32)

    if data_args.use_textual_state_desc and training_args.grid_state_loss_weight != 0.0:
        rank0_print("[warn] use_textual_state_desc=True, forcing grid_state_loss_weight=0.0")
        training_args.grid_state_loss_weight = 0.0

    # Liger kernel
    if training_args.use_liger:
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
            apply_liger_kernel_to_qwen2_vl()
        except ImportError:
            rank0_print("[warn] liger_kernel not available, skipping")

    # --- Load model ---
    model = GridCoTForConditionalGeneration.from_pretrained(
        model_args.model_path,
        torch_dtype=compute_dtype,
        attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
    )
    model.config.use_cache = False

    # Freeze
    set_requires_grad(model.model.parameters(), not training_args.freeze_llm)
    set_requires_grad(model.lm_head.parameters(), not training_args.freeze_llm)
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=training_args.device)
    set_requires_grad(vision_tower.parameters(), not training_args.freeze_vision_tower)
    set_requires_grad(vision_tower.merger.parameters(), training_args.tune_merger)

    # Gradient checkpointing
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # --- LoRA ---
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

    # Keep grid-state alignment pathway trainable (independent of LoRA freeze).
    force_unfreeze_grid_alignment_modules(model)

    # --- Tokenizer setup ---
    processor = AutoProcessor.from_pretrained(model_args.model_path, padding_side="right")
    old_vocab_size = len(processor.tokenizer)
    processor.tokenizer.add_tokens(NEW_TOKENS)
    new_vocab_size = len(processor.tokenizer)

    if new_vocab_size > old_vocab_size:
        rank0_print(f"Tokenizer size after adding tokens: {new_vocab_size}")

        # Gather for ZeRO-3 compatibility
        embed_weight = model.get_input_embeddings().weight
        if hasattr(embed_weight, "ds_id"):
            with zero.GatheredParameters([embed_weight]):
                current_size = embed_weight.data.shape[0]
        else:
            current_size = embed_weight.data.shape[0]

        if new_vocab_size > current_size:
            model.resize_token_embeddings(new_vocab_size)

    # Token indices
    def get_token_id(text):
        return processor.tokenizer(text, add_special_tokens=False).input_ids[0]

    model.set_token_indices(
        grid_visible_token_idx=get_token_id(GRID_VISIBLE_TOKEN),
        grid_pad_token_idx=get_token_id(GRID_PAD_TOKEN),
        think_start_idx=get_token_id(THINK_START),
        think_end_idx=get_token_id(THINK_END),
        answer_start_idx=get_token_id(ANSWER_START),
        answer_end_idx=get_token_id(ANSWER_END),
    )

    model.set_loss_weights(
        grid_visible_token_loss_weight=training_args.grid_visible_token_loss_weight,
        grid_pad_token_loss_weight=training_args.grid_pad_token_loss_weight,
        answer_token_loss_weight=training_args.answer_token_loss_weight,
        answer_span_loss_weight=training_args.answer_span_loss_weight,
        answer_transition_loss_weight=training_args.answer_transition_loss_weight,
        grid_state_loss_weight=training_args.grid_state_loss_weight,
    )

    # Enable gradient for embed_tokens and lm_head (for new tokens only)
    for n, p in model.named_parameters():
        if "embed_tokens" in n or "lm_head" in n:
            p.requires_grad = True

    # Gradient mask: update only the rows for our registered special tokens.
    target_token_ids = sorted({get_token_id(token) for token in NEW_TOKENS})

    def row_mask_hook(grad):
        if grad is None:
            return grad
        row_mask = torch.zeros(grad.shape[0], dtype=grad.dtype, device=grad.device)
        for token_id in target_token_ids:
            if 0 <= token_id < grad.shape[0]:
                row_mask[token_id] = 1.0
        return grad * row_mask.view(-1, 1)

    model.get_input_embeddings().weight.register_hook(row_mask_hook)
    model.get_output_embeddings().weight.register_hook(row_mask_hook)

    # --- Data ---
    data_module = make_data_module(processor=processor, data_args=data_args)
    resume_cb = ResumeDatasetCallback(train_dataset=data_module["train_dataset"])

    # --- Train ---
    trainer = GridCoTTrainer(
        model=model,
        processor=processor,
        trainable_token_ids=target_token_ids,
        args=training_args,
        callbacks=[resume_cb],
        **data_module,
    )

    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    trainer_state_path = None
    if last_checkpoint is not None:
        trainer_state_path = os.path.join(last_checkpoint, "trainer_state.json")

    should_resume = last_checkpoint is not None and os.path.exists(trainer_state_path)
    if should_resume and checkpoint_has_nonfinite_adapter(last_checkpoint):
        rank0_print(
            f"[warn] Found non-finite tensors in {last_checkpoint}; starting fresh instead of resuming."
        )
        should_resume = False

    if should_resume:
        rank0_print(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        if last_checkpoint is not None:
            rank0_print(
                f"[warn] Found incomplete checkpoint at {last_checkpoint} without trainer_state.json; starting fresh."
            )
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True

    # Save
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora = get_peft_state_non_lora_maybe_zero_3(model.named_parameters(), require_grad_only=True)
        if local_rank in (0, -1):
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        if trainer.deepspeed:
            torch.cuda.synchronize()
            trainer.save_model(training_args.output_dir)
        else:
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            trainer._save(training_args.output_dir, state_dict=state_dict)


if __name__ == "__main__":
    train()
