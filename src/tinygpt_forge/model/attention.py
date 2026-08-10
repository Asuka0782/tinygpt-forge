"""Shared-parameter manual and SDPA causal self-attention."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tinygpt_forge.cache import StaticKVCache
from tinygpt_forge.config import ATTENTION_BACKENDS, ModelConfig
from tinygpt_forge.model.rope import RotaryEmbedding

KeyValue = tuple[Tensor, Tensor]


@dataclass
class AttentionOutput:
    """Attention output and optional observability/cache products."""

    hidden_states: Tensor
    probabilities: Tensor | None = None
    present_key_value: KeyValue | None = None


def repeat_key_value(states: Tensor, query_groups: int) -> Tensor:
    """Expand `[B, Hkv, T, D]` key/value states to query-head count."""

    if states.ndim != 4:
        raise ValueError(f"states must have shape [B, Hkv, T, D], got {tuple(states.shape)}")
    if query_groups <= 0:
        raise ValueError("query_groups must be positive")
    if query_groups == 1:
        return states
    return states.repeat_interleave(query_groups, dim=1)


class CausalSelfAttention(nn.Module):
    """Causal MHA/GQA with a readable manual path and a PyTorch SDPA path."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(config.head_dim, config.rope_base)

    def _project(self, x: Tensor, positions: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        batch, time, _ = x.shape
        query = (
            self.q_proj(x)
            .view(batch, time, self.config.n_heads, self.config.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.k_proj(x)
            .view(batch, time, self.config.n_kv_heads, self.config.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(x)
            .view(batch, time, self.config.n_kv_heads, self.config.head_dim)
            .transpose(1, 2)
        )
        query = self.rope(query, positions)
        key = self.rope(key, positions)
        return query, key, value

    def _manual_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        past_length: int,
    ) -> tuple[Tensor, Tensor]:
        scores = query @ key.transpose(-2, -1)
        scores = scores * (1.0 / math.sqrt(self.config.head_dim))
        query_positions = torch.arange(
            past_length,
            past_length + query.size(-2),
            device=query.device,
        )
        key_positions = torch.arange(key.size(-2), device=query.device)
        causal = key_positions[None, :] <= query_positions[:, None]
        scores = scores.masked_fill(~causal, float("-inf"))
        probabilities = F.softmax(scores, dim=-1)
        probabilities = F.dropout(
            probabilities,
            p=self.config.dropout,
            training=self.training,
        )
        return probabilities @ value, probabilities

    def forward(
        self,
        x: Tensor,
        *,
        positions: Tensor | None = None,
        backend: str | None = None,
        return_attention: bool = False,
        past_key_value: KeyValue | None = None,
        use_cache: bool = False,
        static_cache: StaticKVCache | None = None,
        layer_index: int | None = None,
    ) -> AttentionOutput:
        """Return attention states plus optional probabilities and raw GQA K/V cache."""

        if x.ndim != 3 or x.size(-1) != self.config.d_model:
            raise ValueError(
                f"x must have shape [B, T, {self.config.d_model}], got {tuple(x.shape)}"
            )
        selected = backend or self.config.attention_backend
        if selected not in ATTENTION_BACKENDS:
            choices = ", ".join(sorted(ATTENTION_BACKENDS))
            raise ValueError(f"backend must be one of: {choices}")
        if return_attention and selected != "manual":
            raise ValueError("attention probabilities are observable only in the manual backend")

        if past_key_value is not None and static_cache is not None:
            raise ValueError("past_key_value and static_cache are mutually exclusive")
        if static_cache is not None and layer_index is None:
            raise ValueError("layer_index is required with static_cache")

        past_length = static_cache.length if static_cache is not None else 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            expected_prefix = (x.size(0), self.config.n_kv_heads)
            if (
                past_key.ndim != 4
                or past_value.ndim != 4
                or past_key.shape != past_value.shape
                or past_key.shape[:2] != expected_prefix
                or past_key.size(-1) != self.config.head_dim
            ):
                raise ValueError(
                    "past_key_value tensors must both have shape "
                    f"[B, {self.config.n_kv_heads}, S, {self.config.head_dim}]"
                )
            past_length = past_key.size(-2)
        if positions is None:
            positions = torch.arange(
                past_length,
                past_length + x.size(1),
                device=x.device,
            )

        query, raw_key, raw_value = self._project(x, positions)
        if static_cache is not None:
            assert layer_index is not None
            raw_key, raw_value = static_cache.update(
                layer_index,
                raw_key,
                raw_value,
                start_position=past_length,
            )
        elif past_key_value is not None:
            past_key, past_value = past_key_value
            if (
                past_key.device != raw_key.device
                or past_value.device != raw_value.device
                or past_key.dtype != raw_key.dtype
                or past_value.dtype != raw_value.dtype
            ):
                raise ValueError("past_key_value device and dtype must match current key/value")
            raw_key = torch.cat((past_key, raw_key), dim=-2)
            raw_value = torch.cat((past_value, raw_value), dim=-2)
        present_key_value = (raw_key, raw_value) if use_cache and static_cache is None else None

        key = repeat_key_value(raw_key, self.config.query_groups)
        value = repeat_key_value(raw_value, self.config.query_groups)

        probabilities: Tensor | None = None
        if selected == "manual":
            attended, probabilities = self._manual_attention(
                query,
                key,
                value,
                past_length=past_length,
            )
        else:
            dropout_p = self.config.dropout if self.training else 0.0
            query_length = query.size(-2)
            key_length = key.size(-2)
            use_square_causal = past_length == 0 and query_length == key_length
            attention_mask = None
            if not use_square_causal:
                query_positions = torch.arange(
                    past_length,
                    past_length + query_length,
                    device=query.device,
                )
                key_positions = torch.arange(key_length, device=query.device)
                attention_mask = key_positions[None, :] <= query_positions[:, None]
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=use_square_causal,
            )

        batch, _, time, _ = attended.shape
        merged = attended.transpose(1, 2).contiguous().view(batch, time, self.config.d_model)
        return AttentionOutput(
            hidden_states=self.residual_dropout(self.out_proj(merged)),
            probabilities=probabilities,
            present_key_value=present_key_value,
        )
