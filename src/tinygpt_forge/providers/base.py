"""Small interfaces shared by optional external chat providers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class ProviderError(RuntimeError):
    """A sanitized provider failure that never includes credentials or response bodies."""


@dataclass(frozen=True)
class ChatMessage:
    """One validated OpenAI-style chat message."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError("role must be system, user, or assistant")
        if not self.content:
            raise ValueError("message content must not be empty")
        if len(self.content) > 100_000:
            raise ValueError("message content exceeds the 100,000-character safety limit")


class ChatProvider(Protocol):
    """Synchronous provider boundary used by the optional CLI client."""

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Return one assistant message or raise :class:`ProviderError`."""
