"""Custom Trainer with separate LR for projection layers and resume support."""

import os
import torch
import torch.nn as nn
from typing import Optional

from transformers import Trainer, TrainerCallback
from transformers.trainer import (
    get_parameter_names,
    is_peft_available,
    WEIGHTS_NAME,
    TRAINING_ARGS_NAME,
    SAFE_WEIGHTS_NAME,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
)
try:
    from transformers.trainer import ALL_LAYERNORM_LAYERS
except ImportError:
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import safetensors
from peft import PeftModel
from transformers.processing_utils import ProcessorMixin
from transformers.modeling_utils import PreTrainedModel

TRAINABLE_TOKEN_ROWS_NAME = "trainable_token_rows.pt"


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    def maybe_zero_3(param):
        if hasattr(param, "ds_id"):
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                pass
            with zero.GatheredParameters([param]):
                param = param.data.detach().cpu().clone()
        else:
            param = param.detach().cpu().clone()
        return param

    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v).cpu() for k, v in to_return.items()}
    return to_return


def _gather_param(param):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            pass
        with zero.GatheredParameters([param]):
            return param.data.detach().cpu().clone()
    return param.detach().cpu().clone()


def save_trainable_token_rows(model, output_dir, token_ids):
    if not token_ids:
        return

    input_weight = _gather_param(model.get_input_embeddings().weight)
    output_weight = _gather_param(model.get_output_embeddings().weight)

    valid_token_ids = [token_id for token_id in token_ids if 0 <= token_id < input_weight.shape[0]]
    if not valid_token_ids:
        return

    row_state = {
        "token_ids": valid_token_ids,
        "input_embeddings": input_weight[valid_token_ids].clone(),
        "output_embeddings": output_weight[valid_token_ids].clone(),
    }
    torch.save(row_state, os.path.join(output_dir, TRAINABLE_TOKEN_ROWS_NAME))


def load_trainable_token_rows(model, checkpoint_dir):
    path = os.path.join(checkpoint_dir, TRAINABLE_TOKEN_ROWS_NAME)
    if not os.path.exists(path):
        return False

    row_state = torch.load(path, map_location="cpu")
    token_ids = row_state.get("token_ids", [])
    if not token_ids:
        return False

    with torch.no_grad():
        input_weight = model.get_input_embeddings().weight
        output_weight = model.get_output_embeddings().weight

        input_rows = row_state["input_embeddings"].to(device=input_weight.device, dtype=input_weight.dtype)
        output_rows = row_state["output_embeddings"].to(device=output_weight.device, dtype=output_weight.dtype)

        input_weight[token_ids] = input_rows
        output_weight[token_ids] = output_rows
    return True


class GridCoTTrainer(Trainer):

    def __init__(self, processor, trainable_token_ids=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.trainable_token_ids = sorted(set(trainable_token_ids or []))

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        # Separate projection layer parameters for different LR
        projection_parameters = [
            name for name in decay_parameters
            if "_projection" in name or "query_vectors" in name or "cross_attention" in name
        ]

        projection_lr = getattr(self.args, "projection_layer_lr", None)
        vision_lr = getattr(self.args, "vision_lr", None)

        if vision_lr is not None:
            visual_parameters = [
                name for name, _ in opt_model.named_parameters()
                if "visual" in name and "merger" not in name
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in opt_model.named_parameters()
                               if n in decay_parameters and n not in visual_parameters and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters()
                               if n not in decay_parameters and n not in visual_parameters and p.requires_grad],
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters()
                               if n in visual_parameters and n in decay_parameters and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                    "lr": vision_lr,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters()
                               if n in visual_parameters and n not in decay_parameters and p.requires_grad],
                    "weight_decay": 0.0,
                    "lr": vision_lr,
                },
            ]
        elif projection_lr is not None:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in opt_model.named_parameters()
                               if n in decay_parameters and n not in projection_parameters and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters()
                               if n in decay_parameters and n in projection_parameters and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                    "lr": projection_lr,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters()
                               if n not in decay_parameters and n not in projection_parameters and p.requires_grad],
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters()
                               if n not in decay_parameters and n in projection_parameters and p.requires_grad],
                    "weight_decay": 0.0,
                    "lr": projection_lr,
                },
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in opt_model.named_parameters()
                               if n in decay_parameters and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters()
                               if n not in decay_parameters and p.requires_grad],
                    "weight_decay": 0.0,
                },
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def _save_checkpoint(self, model, trial):
        if getattr(self.args, "lora_enable", False):
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            if self.hp_search_backend is None and trial is None:
                self.store_flos()
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            self.save_model(output_dir, _internal_call=True)
            save_trainable_token_rows(self.model, output_dir, self.trainable_token_ids)

            if not self.args.save_only_model:
                self._save_optimizer_and_scheduler(output_dir)
                self._save_rng_state(output_dir)

            if self.args.should_save:
                self.state.stateful_callbacks["TrainerControl"] = self.control.state()
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.should_save:
                self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)
        else:
            super()._save_checkpoint(model, trial)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        loaded = load_trainable_token_rows(self.model, resume_from_checkpoint)
        if loaded:
            logger.info(f"Loaded trainable token rows from {resume_from_checkpoint}")

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,)
        if is_peft_available():
            supported_classes = (PreTrainedModel, PeftModel)

        if isinstance(self.model, supported_classes):
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )
        elif isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
            self.accelerator.unwrap_model(self.model).save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )
        else:
            if state_dict is None:
                state_dict = self.model.state_dict()
            if self.args.save_safetensors:
                safetensors.torch.save_file(state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME))
            else:
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        if self.processor is not None:
            self.processor.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


class ResumeDatasetCallback(TrainerCallback):
    """Sync dataset cur_step when resuming from checkpoint."""

    def __init__(self, train_dataset):
        self.train_dataset = train_dataset
        self._resumed = False

    def on_train_begin(self, args, state, control, **kwargs):
        if state.global_step > 0 and not self._resumed:
            samples_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps
            resumed_step = state.global_step * samples_per_step
            self.train_dataset.set_cur_step(resumed_step)
            self._resumed = True
            print(f"[ResumeDatasetCallback] Dataset cur_step set to {resumed_step}")
