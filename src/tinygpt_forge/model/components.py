"""Normalization and feed-forward components."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Root mean square normalization without mean subtraction."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize the final dimension, accumulating half inputs in float32."""

        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        scale = torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + self.eps)
        normalized = (work * scale).to(dtype=x.dtype)
        return normalized * self.weight


class SwiGLU(nn.Module):
    """Gated feed-forward network using the SiLU activation."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Return ``down(silu(gate(x)) * up(x))``."""

        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return cast(Tensor, self.down_proj(gated))
