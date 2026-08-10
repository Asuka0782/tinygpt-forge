"""Explicit next-token loss conventions."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def aligned_next_token_cross_entropy(
    logits: Tensor,
    next_token_ids: Tensor,
    *,
    ignore_index: int = -100,
) -> Tensor:
    """Score logits against an already shifted next-token target tensor.

    Args:
        logits: Floating-point tensor with shape ``[batch, time, vocab]``.
        next_token_ids: Integer tensor with shape ``[batch, time]``.
        ignore_index: Target value excluded from the loss.
    """

    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, T, V], got {tuple(logits.shape)}")
    if next_token_ids.shape != logits.shape[:2]:
        raise ValueError(
            "next_token_ids must match logits [B, T], "
            f"got {tuple(next_token_ids.shape)} and {tuple(logits.shape)}"
        )
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        next_token_ids.reshape(-1),
        ignore_index=ignore_index,
    )


def shifted_next_token_cross_entropy(
    logits: Tensor,
    token_ids: Tensor,
    *,
    ignore_index: int = -100,
) -> Tensor:
    """Apply the teacher-forced one-token shift to one complete token sequence."""

    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, T, V], got {tuple(logits.shape)}")
    if token_ids.shape != logits.shape[:2]:
        raise ValueError(f"token_ids must match logits [B, T], got {tuple(token_ids.shape)}")
    if logits.size(1) < 2:
        raise ValueError("shifted loss needs a sequence length of at least two")
    return aligned_next_token_cross_entropy(
        logits[:, :-1, :],
        token_ids[:, 1:],
        ignore_index=ignore_index,
    )
