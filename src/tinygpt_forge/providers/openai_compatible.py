"""A dependency-free, explicitly configured OpenAI-compatible chat client."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

from tinygpt_forge.providers.base import ChatMessage, ProviderError

MAX_PROVIDER_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_MESSAGE_CHARACTERS = 1_000_000


@dataclass(frozen=True)
class TransportResponse:
    """Minimal HTTP response consumed by the provider."""

    status: int
    body: bytes


class ProviderConnectionError(RuntimeError):
    """Internal retryable transport error without credential-bearing details."""


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects so bearer credentials never cross an endpoint boundary."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> None:
        del request, file_pointer, code, message, headers, new_url
        return None


class HTTPTransport(Protocol):
    """Injectable network boundary so tests never call a paid endpoint."""

    def send(
        self,
        request: urllib.request.Request,
        *,
        timeout_seconds: float,
        proxy_url: str | None,
    ) -> TransportResponse:
        """Send a request and return status/body without interpreting provider JSON."""


class UrllibTransport:
    """Standard-library HTTP transport with optional explicit proxy configuration."""

    def send(
        self,
        request: urllib.request.Request,
        *,
        timeout_seconds: float,
        proxy_url: str | None,
    ) -> TransportResponse:
        proxies = {} if proxy_url is None else {"http": proxy_url, "https": proxy_url}
        handlers: list[urllib.request.BaseHandler] = [
            urllib.request.ProxyHandler(proxies),
            _NoRedirectHandler(),
        ]
        opener = urllib.request.build_opener(*handlers)
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                return TransportResponse(
                    status=response.status,
                    body=response.read(MAX_PROVIDER_RESPONSE_BYTES + 1),
                )
        except urllib.error.HTTPError as error:
            return TransportResponse(
                status=error.code,
                body=error.read(MAX_PROVIDER_RESPONSE_BYTES + 1),
            )
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ProviderConnectionError("provider connection failed") from error


def _validated_url(value: str, *, field_name: str, allow_local_http: bool) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{field_name} must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{field_name} must not contain embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{field_name} must not contain a query string or fragment")
    local_hosts = {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme == "http" and (not allow_local_http or parsed.hostname not in local_hosts):
        raise ValueError(f"{field_name} requires HTTPS except for localhost")
    return value.rstrip("/")


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    """External endpoint settings loaded only from explicit inputs/environment."""

    base_url: str
    model: str
    api_key: str = field(repr=False)
    timeout_seconds: float = 30.0
    max_retries: int = 2
    proxy_url: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_url",
            _validated_url(self.base_url, field_name="base_url", allow_local_http=True),
        )
        if not self.model or len(self.model) > 256:
            raise ValueError("model must contain 1 to 256 characters")
        if not self.api_key or len(self.api_key) > 4096:
            raise ValueError("api_key must contain 1 to 4096 characters")
        if any(ord(character) < 32 or ord(character) == 127 for character in self.api_key):
            raise ValueError("api_key must not contain ASCII control characters")
        if not 0 < self.timeout_seconds <= 300:
            raise ValueError("timeout_seconds must satisfy 0 < timeout <= 300")
        if not 0 <= self.max_retries <= 5:
            raise ValueError("max_retries must satisfy 0 <= retries <= 5")
        if self.proxy_url is not None:
            object.__setattr__(
                self,
                "proxy_url",
                _validated_url(
                    self.proxy_url,
                    field_name="proxy_url",
                    allow_local_http=True,
                ),
            )

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> OpenAICompatibleConfig:
        """Read namespaced variables without loading or printing arbitrary process state."""

        source = os.environ if environment is None else environment
        missing = [
            name
            for name in ("TINYGPT_API_KEY", "TINYGPT_BASE_URL", "TINYGPT_MODEL")
            if not source.get(name)
        ]
        if missing:
            raise ValueError(f"missing required environment variables: {', '.join(missing)}")
        return cls(
            api_key=source["TINYGPT_API_KEY"],
            base_url=source["TINYGPT_BASE_URL"],
            model=source["TINYGPT_MODEL"],
            timeout_seconds=float(source.get("TINYGPT_TIMEOUT_SECONDS", "30")),
            max_retries=int(source.get("TINYGPT_MAX_RETRIES", "2")),
            proxy_url=source.get("TINYGPT_PROXY") or None,
        )


class OpenAICompatibleProvider:
    """Call `/chat/completions` with bounded retries and sanitized errors."""

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        transport: HTTPTransport | None = None,
    ) -> None:
        self.config = config
        self.transport = transport or UrllibTransport()

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Return assistant text; never include API keys or provider bodies in errors."""

        if not messages:
            raise ValueError("messages must not be empty")
        if len(messages) > 128:
            raise ValueError("messages exceed the 128-message safety limit")
        if sum(len(message.content) for message in messages) > MAX_TOTAL_MESSAGE_CHARACTERS:
            raise ValueError("total message content exceeds the 1,000,000-character safety limit")
        if not 1 <= max_tokens <= 8192:
            raise ValueError("max_tokens must satisfy 1 <= max_tokens <= 8192")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must satisfy 0 <= temperature <= 2")

        payload = json.dumps(
            {
                "model": self.config.model,
                "messages": [
                    {"role": message.role, "content": message.content} for message in messages
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "tinygpt-forge/0.0.1",
            },
        )

        last_status: int | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.transport.send(
                    request,
                    timeout_seconds=self.config.timeout_seconds,
                    proxy_url=self.config.proxy_url,
                )
            except ProviderConnectionError as error:
                if attempt == self.config.max_retries:
                    raise ProviderError("provider connection failed after retries") from error
            else:
                last_status = response.status
                if len(response.body) > MAX_PROVIDER_RESPONSE_BYTES:
                    raise ProviderError("provider response exceeds the 2 MiB safety limit")
                if 200 <= response.status < 300:
                    return self._parse_content(response.body)
                retryable = response.status in {408, 409, 429} or response.status >= 500
                if not retryable or attempt == self.config.max_retries:
                    raise ProviderError(f"provider request failed with HTTP {response.status}")
            if attempt < self.config.max_retries:
                time.sleep(min(0.25 * (2**attempt), 2.0))

        raise ProviderError(f"provider request failed with HTTP {last_status}")

    @staticmethod
    def _parse_content(body: bytes) -> str:
        try:
            document = json.loads(body.decode("utf-8"))
            content = document["choices"][0]["message"]["content"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            raise ProviderError("provider returned an invalid chat-completions response") from error
        if not isinstance(content, str) or not content:
            raise ProviderError("provider returned an empty assistant message")
        return content
