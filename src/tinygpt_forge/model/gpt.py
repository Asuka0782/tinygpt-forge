"""Modern, small decoder-only Transformer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tinygpt_forge.cache import StaticKVCache
from tinygpt_forge.config import ModelConfig
from tinygpt_forge.losses import aligned_next_token_cross_entropy
from tinygpt_forge.model.attention import CausalSelfAttention, KeyValue
from tinygpt_forge.model.components import RMSNorm, SwiGLU


@dataclass
class CausalLMOutput:
    """Structured model output with stable tensor meanings."""

    logits: Tensor
    loss: Tensor | None = None
    attentions: tuple[Tensor, ...] | None = None
    past_key_values: tuple[KeyValue, ...] | None = None


class DecoderBlock(nn.Module):
    """Pre-normalized attention and SwiGLU residual block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.feed_forward = SwiGLU(config.d_model, config.d_ff)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        *,
        backend: str | None = None,
        return_attention: bool = False,
        past_key_value: KeyValue | None = None,
        use_cache: bool = False,
        static_cache: StaticKVCache | None = None,
        layer_index: int | None = None,
    ) -> tuple[Tensor, Tensor | None, KeyValue | None]:
        """Apply both pre-norm residual sublayers."""

        attention_output = self.attention(
            self.attention_norm(x),
            backend=backend,
            return_attention=return_attention,
            past_key_value=past_key_value,
            use_cache=use_cache,
            static_cache=static_cache,
            layer_index=layer_index,
        )
        x = x + attention_output.hidden_states
        x = x + self.residual_dropout(self.feed_forward(self.ffn_norm(x)))
        return x, attention_output.probabilities, attention_output.present_key_value


class TinyGPT(nn.Module):
    """Decoder-only language model shared by the teaching and SDPA paths."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        *,
        targets: Tensor | None = None,
        backend: str | None = None,
        return_attentions: bool = False,
        past_key_values: tuple[KeyValue | None, ...] | None = None,
        use_cache: bool = False,
        static_cache: StaticKVCache | None = None,
    ) -> CausalLMOutput:
        """Compute logits and optional loss against already shifted targets.

        ``targets[b, t]`` is the token that should follow ``input_ids[b, t]``.
        Use :func:`shifted_next_token_cross_entropy` when passing one unshifted sequence.
        """

        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must use torch.int32 or torch.int64")
        if input_ids.size(1) == 0:
            raise ValueError("input_ids must contain at least one token")

        if past_key_values is not None and static_cache is not None:
            raise ValueError("past_key_values and static_cache are mutually exclusive")
        if static_cache is not None:
            if not use_cache:
                raise ValueError("static_cache requires use_cache=True")
            if static_cache.config != self.config:
                raise ValueError("static cache configuration does not match the model")
            if static_cache.batch_size != input_ids.size(0):
                raise ValueError("static cache batch size does not match input_ids")

        if past_key_values is None:
            layer_past: tuple[KeyValue | None, ...] = (None,) * self.config.n_layers
        else:
            if len(past_key_values) != self.config.n_layers:
                raise ValueError(
                    f"past_key_values must contain {self.config.n_layers} layer entries"
                )
            layer_past = past_key_values
        past_lengths = {key_value[0].size(-2) for key_value in layer_past if key_value is not None}
        if len(past_lengths) > 1:
            raise ValueError("all cached layers must have the same sequence length")
        dynamic_past_length = next(iter(past_lengths), 0)
        past_length = static_cache.length if static_cache is not None else dynamic_past_length
        if (
            static_cache is None
            and any(key_value is None for key_value in layer_past)
            and past_length != 0
        ):
            raise ValueError("cached and uncached layers cannot be mixed")

        total_length = past_length + input_ids.size(1)
        if total_length > self.config.max_seq_len:
            raise ValueError(
                f"sequence length with cache {total_length} exceeds max_seq_len "
                f"{self.config.max_seq_len}"
            )

        hidden = self.embedding_dropout(self.token_embedding(input_ids))
        observed: list[Tensor] | None = [] if return_attentions else None
        present: list[KeyValue] | None = [] if use_cache and static_cache is None else None
        for layer_index, (block, block_past) in enumerate(
            zip(self.blocks, layer_past, strict=True)
        ):
            hidden, probabilities, block_present = block(
                hidden,
                backend=backend,
                return_attention=return_attentions,
                past_key_value=block_past,
                use_cache=use_cache,
                static_cache=static_cache,
                layer_index=layer_index,
            )
            if observed is not None:
                if probabilities is None:
                    raise RuntimeError("manual attention did not return its probabilities")
                observed.append(probabilities)
            if present is not None:
                if block_present is None:
                    raise RuntimeError("cache-enabled attention did not return key/value tensors")
                present.append(block_present)

        if static_cache is not None:
            static_cache.advance(input_ids.size(1))

        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = aligned_next_token_cross_entropy(logits, targets)
        attentions = tuple(observed) if observed is not None else None
        cache = tuple(present) if present is not None else None
        return CausalLMOutput(
            logits=logits,
            loss=loss,
            attentions=attentions,
            past_key_values=cache,
        )

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        """Count unique parameter elements, respecting tied embeddings."""

        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)
