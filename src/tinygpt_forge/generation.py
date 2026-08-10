"""Autoregressive generation with equivalent full and cached decode paths."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from tinygpt_forge.cache import StaticKVCache
from tinygpt_forge.model.attention import KeyValue
from tinygpt_forge.model.gpt import TinyGPT


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling controls with deterministic seed handling."""

    max_new_tokens: int = 32
    temperature: float = 0.0
    top_k: int | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive when provided")


def sample_next_token(
    logits: Tensor, config: GenerationConfig, generator: torch.Generator
) -> Tensor:
    """Sample one token from `[B, V]` logits, or greedily select at temperature zero."""

    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B, V], got {tuple(logits.shape)}")
    if config.temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    scaled = logits / config.temperature
    if config.top_k is not None:
        k = min(config.top_k, scaled.size(-1))
        threshold = torch.topk(scaled, k=k, dim=-1).values[:, -1, None]
        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
    probabilities = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


@torch.inference_mode()
def generate(
    model: TinyGPT,
    input_ids: Tensor,
    config: GenerationConfig,
    *,
    backend: str | None = None,
    use_cache: bool = True,
    cache_implementation: str = "dynamic",
) -> Tensor:
    """Generate tokens while preserving full/cached sampling semantics."""

    if input_ids.ndim != 2 or input_ids.size(1) == 0:
        raise ValueError("input_ids must have non-empty shape [B, T]")
    if input_ids.size(1) + config.max_new_tokens > model.config.max_seq_len:
        raise ValueError("prompt plus max_new_tokens exceeds the configured context length")
    if cache_implementation not in {"dynamic", "static"}:
        raise ValueError("cache_implementation must be dynamic or static")

    generated = input_ids
    generator = torch.Generator(device=input_ids.device)
    generator.manual_seed(config.seed)
    past_key_values: tuple[KeyValue, ...] | None = None
    static_cache = None
    if use_cache and cache_implementation == "static":
        parameter = next(model.parameters())
        static_cache = StaticKVCache(
            model.config,
            batch_size=input_ids.size(0),
            capacity=input_ids.size(1) + config.max_new_tokens,
            device=input_ids.device,
            dtype=parameter.dtype,
        )
    step_input = generated

    for _ in range(config.max_new_tokens):
        if use_cache:
            output = model(
                step_input,
                backend=backend,
                past_key_values=past_key_values,
                use_cache=True,
                static_cache=static_cache,
            )
            if static_cache is None:
                if output.past_key_values is None:
                    raise RuntimeError("dynamic-cache model call did not return key/value tensors")
                past_key_values = output.past_key_values
        else:
            output = model(generated, backend=backend)

        next_token = sample_next_token(output.logits[:, -1, :], config, generator)
        generated = torch.cat((generated, next_token), dim=1)
        step_input = next_token

    return generated
