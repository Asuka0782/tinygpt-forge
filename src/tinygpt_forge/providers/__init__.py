"""Optional provider abstractions; the core project remains local-first."""

from tinygpt_forge.providers.base import ChatMessage, ChatProvider, ProviderError
from tinygpt_forge.providers.openai_compatible import (
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
)

__all__ = [
    "ChatMessage",
    "ChatProvider",
    "OpenAICompatibleConfig",
    "OpenAICompatibleProvider",
    "ProviderError",
]
