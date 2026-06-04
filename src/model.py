"""
GridCoT model: Qwen2.5-VL with weighted CE loss for structured CoT generation.

Adds token-level loss weighting and grid-state supervision
to supervise <grid_token> positions with checkpoint state targets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

from constants import IGNORE_INDEX


class GridCoTForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """Qwen2.5-VL with token-weighted CE loss for grid CoT training."""

    def __init__(self, config):
        super().__init__(config)

        # Token indices (set after tokenizer setup)
        self.grid_visible_token_idx = None
        self.grid_pad_token_idx = None
        self.think_start_idx = None
        self.think_end_idx = None
        self.answer_start_idx = None
        self.answer_end_idx = None

        # Loss weights
        self.grid_visible_token_loss_weight = 1.0
        self.grid_pad_token_loss_weight = 1.0
        self.answer_token_loss_weight = 1.0
        self.answer_span_loss_weight = 1.0
        self.answer_transition_loss_weight = 1.0
        self.grid_state_loss_weight = 0.0

        self.grid_state_encoder = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, config.hidden_size),
        )
        self.grid_hidden_projector = nn.Linear(config.hidden_size, config.hidden_size)
        self.grid_state_struct_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
        )
        self.grid_state_row_head = nn.Linear(config.hidden_size, 1)
        self.grid_state_col_head = nn.Linear(config.hidden_size, 1)
        self.grid_state_dir_head = nn.Linear(config.hidden_size, 4)
        self.grid_state_key_head = nn.Linear(config.hidden_size, 1)
        self.grid_state_door_head = nn.Linear(config.hidden_size, 1)

    def set_token_indices(
        self,
        grid_visible_token_idx=None,
        grid_pad_token_idx=None,
        think_start_idx=None,
        think_end_idx=None,
        answer_start_idx=None,
        answer_end_idx=None,
    ):
        self.grid_visible_token_idx = grid_visible_token_idx
        self.grid_pad_token_idx = grid_pad_token_idx
        self.think_start_idx = think_start_idx
        self.think_end_idx = think_end_idx
        self.answer_start_idx = answer_start_idx
        self.answer_end_idx = answer_end_idx

    def set_loss_weights(
        self,
        grid_visible_token_loss_weight=1.0,
        grid_pad_token_loss_weight=1.0,
        answer_token_loss_weight=1.0,
        answer_span_loss_weight=1.0,
        answer_transition_loss_weight=1.0,
        grid_state_loss_weight=0.0,
    ):
        self.grid_visible_token_loss_weight = grid_visible_token_loss_weight
        self.grid_pad_token_loss_weight = grid_pad_token_loss_weight
        self.answer_token_loss_weight = answer_token_loss_weight
        self.answer_span_loss_weight = answer_span_loss_weight
        self.answer_transition_loss_weight = answer_transition_loss_weight
        self.grid_state_loss_weight = grid_state_loss_weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        cache_position=None,
        second_per_grid_ts=None,
        # Ignored keys from data collator
        grid_images=None,
        grid_state_targets=None,
        image_files=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        needs_hidden_states = output_hidden_states or (
            grid_state_targets is not None and self.grid_state_loss_weight > 0
        )

        # Forward through the base model (handles vision encoding internally)
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,  # We compute loss ourselves
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=needs_hidden_states,
            return_dict=True,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
        )

        logits = outputs.logits
        loss = None

        if labels is not None:
            loss = self._compute_weighted_ce_loss(logits, labels)
            if grid_state_targets is not None and self.grid_state_loss_weight > 0:
                state_loss = self._compute_grid_state_alignment_loss(
                    hidden_states=outputs.hidden_states[-1],
                    input_ids=input_ids,
                    labels=labels,
                    grid_state_targets=grid_state_targets,
                )
                loss = loss + self.grid_state_loss_weight * state_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _compute_grid_state_alignment_loss(self, hidden_states, input_ids, labels, grid_state_targets):
        losses = []
        batch_size = input_ids.shape[0]

        for batch_idx in range(batch_size):
            sample_targets = grid_state_targets[batch_idx] if batch_idx < len(grid_state_targets) else []
            if not sample_targets:
                continue

            sample_input_ids = input_ids[batch_idx]
            sample_labels = labels[batch_idx]
            sample_hidden = hidden_states[batch_idx]

            visible_mask = sample_input_ids.eq(self.grid_visible_token_idx)
            if sample_labels is not None:
                visible_mask = visible_mask & sample_labels.ne(IGNORE_INDEX)
            token_positions = visible_mask.nonzero(as_tuple=False).view(-1).tolist()

            if len(token_positions) != len(sample_targets):
                min_len = min(len(token_positions), len(sample_targets))
                token_positions = token_positions[:min_len]
                sample_targets = sample_targets[:min_len]

            for token_pos, state_tensor in zip(token_positions, sample_targets):
                hidden_vec = sample_hidden[token_pos]  # keep original dtype (bf16)

                if isinstance(state_tensor, dict) and state_tensor.get("structured") is not None:
                    losses.append(self._compute_structured_state_loss(hidden_vec.float(), state_tensor["structured"]))
                    continue

                # Dense state-tensor alignment — stay in bf16 for linear ops,
                # use float32 for normalization and loss to avoid nan.
                hidden_vec = self.grid_hidden_projector(hidden_vec)
                if isinstance(state_tensor, dict):
                    state_tensor = state_tensor.get("grid", None)
                if state_tensor is None:
                    continue

                enc_param = next(self.grid_state_encoder.parameters())
                state_tensor = state_tensor.unsqueeze(0).to(
                    device=enc_param.device, dtype=enc_param.dtype
                )
                state_vec = self.grid_state_encoder(state_tensor).squeeze(0)
                # float32 for layer_norm + mse to avoid bf16 overflow
                hidden_vec = F.layer_norm(hidden_vec.float(), hidden_vec.shape)
                state_vec = F.layer_norm(state_vec.float(), state_vec.shape)
                losses.append(F.mse_loss(hidden_vec, state_vec))

        if not losses:
            return hidden_states.new_zeros(())
        return torch.stack(losses).mean()

    def _compute_structured_state_loss(self, hidden_vec, structured_target):
        head_param = next(self.grid_state_struct_head.parameters())
        hidden_vec = hidden_vec.to(dtype=head_param.dtype)
        hidden_vec = self.grid_state_struct_head(hidden_vec)

        row_target = hidden_vec.new_tensor([structured_target["agent_row_norm"]], dtype=torch.float32)
        col_target = hidden_vec.new_tensor([structured_target["agent_col_norm"]], dtype=torch.float32)
        dir_target = hidden_vec.new_tensor([structured_target["agent_dir"]], dtype=torch.long)
        key_target = hidden_vec.new_tensor([structured_target["has_key"]], dtype=torch.float32)
        door_target = hidden_vec.new_tensor([structured_target["door_open"]], dtype=torch.float32)

        row_pred = torch.sigmoid(self.grid_state_row_head(hidden_vec)).view(-1).float()
        col_pred = torch.sigmoid(self.grid_state_col_head(hidden_vec)).view(-1).float()
        dir_logits = self.grid_state_dir_head(hidden_vec).view(1, -1).float()
        key_logits = self.grid_state_key_head(hidden_vec).view(-1).float()
        door_logits = self.grid_state_door_head(hidden_vec).view(-1).float()

        row_loss = F.mse_loss(row_pred, row_target)
        col_loss = F.mse_loss(col_pred, col_target)
        dir_loss = F.cross_entropy(dir_logits, dir_target)
        key_loss = F.binary_cross_entropy_with_logits(key_logits, key_target)
        door_loss = F.binary_cross_entropy_with_logits(door_logits, door_target)

        return row_loss + col_loss + dir_loss + key_loss + door_loss

    def _compute_weighted_ce_loss(self, logits, labels):
        """Compute CE loss with per-token weighting for structural tokens."""
        logits = logits.float()

        # Shift: tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
        shift_labels_2d = labels[..., 1:].contiguous()
        shift_labels = shift_labels_2d.view(-1).to(shift_logits.device)

        # Per-token CE loss
        loss_fct = CrossEntropyLoss(reduction="none")
        token_loss = loss_fct(shift_logits, shift_labels)

        valid_mask = shift_labels.ne(IGNORE_INDEX)

        # Check if any weighting is needed
        needs_weighting = (
            (self.grid_visible_token_idx is not None and self.grid_visible_token_loss_weight != 1.0)
            or (self.grid_pad_token_idx is not None and self.grid_pad_token_loss_weight != 1.0)
            or (self.answer_start_idx is not None and self.answer_token_loss_weight != 1.0)
            or (self.answer_span_loss_weight != 1.0)
            or (self.answer_transition_loss_weight != 1.0)
        )

        if not needs_weighting:
            valid_token_loss = token_loss.masked_select(valid_mask)
            return valid_token_loss.mean() if valid_token_loss.numel() > 0 else token_loss.new_zeros(())

        # Build token weights
        token_weights = valid_mask.to(token_loss.dtype)

        # Weight <grid_token> positions
        if self.grid_visible_token_idx is not None and self.grid_visible_token_loss_weight != 1.0:
            mask = shift_labels.eq(self.grid_visible_token_idx)
            token_weights = token_weights + mask.to(token_loss.dtype) * (self.grid_visible_token_loss_weight - 1.0)

        # Weight <|grid_pad|> positions
        if self.grid_pad_token_idx is not None and self.grid_pad_token_loss_weight != 1.0:
            mask = shift_labels.eq(self.grid_pad_token_idx)
            token_weights = token_weights + mask.to(token_loss.dtype) * (self.grid_pad_token_loss_weight - 1.0)

        # Weight <answer> and </answer> tag tokens
        if self.answer_start_idx is not None and self.answer_end_idx is not None and self.answer_token_loss_weight != 1.0:
            mask = shift_labels.eq(self.answer_start_idx) | shift_labels.eq(self.answer_end_idx)
            token_weights = token_weights + mask.to(token_loss.dtype) * (self.answer_token_loss_weight - 1.0)

        # Weight entire answer span (<answer>..content..</answer>)
        if self.answer_start_idx is not None and self.answer_end_idx is not None and self.answer_span_loss_weight != 1.0:
            valid_mask_2d = shift_labels_2d.ne(IGNORE_INDEX)
            start_mask = shift_labels_2d.eq(self.answer_start_idx)
            end_mask = shift_labels_2d.eq(self.answer_end_idx)
            inside = start_mask.long().cumsum(dim=-1) > end_mask.long().cumsum(dim=-1)
            span_mask = (inside | start_mask | end_mask) & valid_mask_2d
            token_weights = token_weights + span_mask.view(-1).to(token_loss.dtype) * (self.answer_span_loss_weight - 1.0)

        # Weight </think> and <answer> transition tokens
        if self.answer_transition_loss_weight != 1.0:
            transition_mask = torch.zeros_like(valid_mask)
            if self.think_end_idx is not None:
                transition_mask = transition_mask | shift_labels.eq(self.think_end_idx)
            if self.answer_start_idx is not None:
                transition_mask = transition_mask | shift_labels.eq(self.answer_start_idx)
            transition_mask = transition_mask & valid_mask
            token_weights = token_weights + transition_mask.to(token_loss.dtype) * (self.answer_transition_loss_weight - 1.0)

        return (token_loss * token_weights).sum() / token_weights.sum().clamp_min(1.0)
