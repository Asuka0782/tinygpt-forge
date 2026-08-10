"""Leakage-conscious token splitting and deterministic next-token batches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TokenSplits:
    """Contiguous train, validation, and test token sequences."""

    train: Tensor
    validation: Tensor
    test: Tensor


@dataclass(frozen=True)
class TextSplits:
    """Contiguous raw-text splits made before fitting a tokenizer."""

    train: str
    validation: str
    test: str


def _split_counts(
    count: int,
    *,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[int, int, int]:
    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("split fractions must satisfy 0 <= fraction < 1")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than one")
    if count < 2:
        raise ValueError("at least two tokens are required")

    validation_count = int(count * validation_fraction)
    test_count = int(count * test_fraction)
    train_count = count - validation_count - test_count
    split_counts = {
        "train": train_count,
        "validation": validation_count,
        "test": test_count,
    }
    fractions = {
        "train": 1 - validation_fraction - test_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }
    for name, split_count in split_counts.items():
        if fractions[name] > 0 and split_count < 2:
            raise ValueError(f"{name} split needs at least two tokens for next-token scoring")
    return train_count, validation_count, test_count


def split_text(
    text: str,
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> TextSplits:
    """Split raw code points before vocabulary fitting to prevent tokenizer leakage."""

    if not text:
        raise ValueError("text must not be empty")
    train_count, validation_count, _ = _split_counts(
        len(text),
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    train_end = train_count
    validation_end = train_count + validation_count
    return TextSplits(
        train=text[:train_end],
        validation=text[train_end:validation_end],
        test=text[validation_end:],
    )


def split_token_ids(
    token_ids: Sequence[int],
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> TokenSplits:
    """Make deterministic contiguous splits without fitting on validation/test tokens."""

    train_count, validation_count, _ = _split_counts(
        len(token_ids),
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    tokens = torch.as_tensor(token_ids, dtype=torch.long)
    train_end = train_count
    validation_end = train_count + validation_count
    return TokenSplits(
        train=tokens[:train_end].clone(),
        validation=tokens[train_end:validation_end].clone(),
        test=tokens[validation_end:].clone(),
    )


class NextTokenBatcher:
    """Sample reproducible `(input, next-token target)` windows from one split."""

    def __init__(
        self,
        token_ids: Tensor,
        *,
        block_size: int,
        batch_size: int,
        seed: int,
        device: torch.device | str = "cpu",
    ) -> None:
        if token_ids.ndim != 1 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be a rank-one torch.long tensor")
        if block_size <= 0 or batch_size <= 0:
            raise ValueError("block_size and batch_size must be positive")
        if token_ids.numel() <= block_size:
            raise ValueError("token sequence must be longer than block_size")
        self.token_ids = token_ids.cpu()
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def next_batch(self) -> tuple[Tensor, Tensor]:
        """Return tensors with shape `[batch_size, block_size]`."""

        max_start = self.token_ids.numel() - self.block_size
        starts = torch.randint(
            0,
            max_start,
            (self.batch_size,),
            generator=self.generator,
        )
        inputs = torch.stack(
            [self.token_ids[start : start + self.block_size] for start in starts.tolist()]
        )
        targets = torch.stack(
            [self.token_ids[start + 1 : start + self.block_size + 1] for start in starts.tolist()]
        )
        return inputs.to(self.device), targets.to(self.device)

    def get_rng_state(self) -> Tensor:
        """Return a clone of the batch sampler RNG state for exact resume."""

        return self.generator.get_state().clone()

    def set_rng_state(self, state: Tensor) -> None:
        """Restore a state produced by :meth:`get_rng_state`."""

        self.generator.set_state(state.cpu())
