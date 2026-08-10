"""Pairwise rotary position embeddings (RoPE)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    """Apply pairwise rotations to the last dimension of `[B, H, T, D]`."""

    inv_freq: Tensor

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even integer")
        if base <= 0:
            raise ValueError("base must be positive")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
        self.head_dim = head_dim
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        """Rotate ``x`` at the supplied absolute positions."""

        if x.ndim != 4 or x.size(-1) != self.head_dim:
            raise ValueError(f"x must have shape [B, H, T, {self.head_dim}], got {tuple(x.shape)}")
        time = x.size(-2)
        if positions is None:
            positions = torch.arange(time, device=x.device)
        if positions.ndim != 1 or positions.numel() != time:
            raise ValueError(f"positions must have shape [{time}], got {tuple(positions.shape)}")

        angle_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        angles = (
            positions.to(device=x.device, dtype=angle_dtype)[:, None]
            * self.inv_freq.to(device=x.device, dtype=angle_dtype)[None, :]
        )
        cos = angles.cos().to(dtype=x.dtype)[None, None, :, :]
        sin = angles.sin().to(dtype=x.dtype)[None, None, :, :]

        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
