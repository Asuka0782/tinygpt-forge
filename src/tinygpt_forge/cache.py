"""Preallocated key/value cache storage for allocation-stable decoding."""

from __future__ import annotations

import torch
from torch import Tensor

from tinygpt_forge.config import ModelConfig


class StaticKVCache:
    """Own fixed-capacity, unexpanded GQA key/value tensors for every layer.

    The cache stores `[batch, n_kv_heads, capacity, head_dim]` rather than repeated query
    heads. One model call writes the same position range in every layer, then advances the
    shared logical length exactly once.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        batch_size: int,
        capacity: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 < capacity <= config.max_seq_len:
            raise ValueError("capacity must satisfy 0 < capacity <= max_seq_len")
        if not dtype.is_floating_point:
            raise ValueError("cache dtype must be floating point")
        self.config = config
        self.batch_size = batch_size
        self.capacity = capacity
        requested_device = torch.device(device)
        self.dtype = dtype
        shape = (batch_size, config.n_kv_heads, capacity, config.head_dim)
        self._keys = [
            torch.empty(shape, device=requested_device, dtype=dtype) for _ in range(config.n_layers)
        ]
        self.device = self._keys[0].device
        self._values = [
            torch.empty(shape, device=self.device, dtype=dtype) for _ in range(config.n_layers)
        ]
        self.length = 0

    def update(
        self,
        layer_index: int,
        key: Tensor,
        value: Tensor,
        *,
        start_position: int,
    ) -> tuple[Tensor, Tensor]:
        """Write one layer's new states and return views through the written end."""

        if not 0 <= layer_index < self.config.n_layers:
            raise ValueError(f"layer_index out of range: {layer_index}")
        if start_position != self.length:
            raise ValueError(
                f"start_position {start_position} does not match cache length {self.length}"
            )
        expected_prefix = (self.batch_size, self.config.n_kv_heads)
        if (
            key.ndim != 4
            or value.shape != key.shape
            or key.shape[:2] != expected_prefix
            or key.size(-1) != self.config.head_dim
            or key.device != self.device
            or value.device != self.device
            or key.dtype != self.dtype
            or value.dtype != self.dtype
        ):
            raise ValueError(
                "new key/value tensors do not match the static cache batch, heads, device, or dtype"
            )
        end_position = start_position + key.size(-2)
        if end_position > self.capacity:
            raise ValueError(f"cache write through {end_position} exceeds capacity {self.capacity}")
        self._keys[layer_index][:, :, start_position:end_position, :].copy_(key)
        self._values[layer_index][:, :, start_position:end_position, :].copy_(value)
        return (
            self._keys[layer_index][:, :, :end_position, :],
            self._values[layer_index][:, :, :end_position, :],
        )

    def advance(self, token_count: int) -> None:
        """Commit one completed model call after all layers wrote their states."""

        if token_count <= 0:
            raise ValueError("token_count must be positive")
        if self.length + token_count > self.capacity:
            raise ValueError("cache advance exceeds capacity")
        self.length += token_count

    def reset(self) -> None:
        """Reset logical length without reallocating or clearing inaccessible old values."""

        self.length = 0

    def rewind(self, length: int) -> None:
        """Move logical length backward while retaining reusable prefix storage."""

        if not 0 <= length <= self.length:
            raise ValueError(f"rewind length must satisfy 0 <= length <= {self.length}")
        self.length = length

    @property
    def allocated_bytes(self) -> int:
        """Return storage bytes for all key and value tensors."""

        return sum(
            tensor.numel() * tensor.element_size() for tensor in (*self._keys, *self._values)
        )

    def layer_storage(self, layer_index: int) -> tuple[Tensor, Tensor]:
        """Expose full-capacity storage for tests and memory inspection."""

        if not 0 <= layer_index < self.config.n_layers:
            raise ValueError(f"layer_index out of range: {layer_index}")
        return self._keys[layer_index], self._values[layer_index]
