from __future__ import annotations

import json
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from tinygpt_forge.providers import (
    ChatMessage,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    ProviderError,
)
from tinygpt_forge.providers.openai_compatible import (
    MAX_PROVIDER_RESPONSE_BYTES,
    TransportResponse,
    UrllibTransport,
    _NoRedirectHandler,
)


class FakeTransport:
    def __init__(self, responses: list[TransportResponse]) -> None:
        self.responses = responses
        self.requests: list[urllib.request.Request] = []

    def send(
        self,
        request: urllib.request.Request,
        *,
        timeout_seconds: float,
        proxy_url: str | None,
    ) -> TransportResponse:
        del timeout_seconds, proxy_url
        self.requests.append(request)
        return self.responses.pop(0)


class OpenAICompatibleProviderTests(unittest.TestCase):
    def test_standard_transport_uses_only_the_explicit_proxy(self) -> None:
        request = urllib.request.Request("https://provider.example/v1/chat/completions")
        opener = unittest.mock.MagicMock()
        opener.open.side_effect = urllib.error.URLError("offline")
        with patch(
            "tinygpt_forge.providers.openai_compatible.urllib.request.build_opener",
            return_value=opener,
        ) as build_opener:
            with self.assertRaisesRegex(RuntimeError, "connection failed"):
                UrllibTransport().send(request, timeout_seconds=1.0, proxy_url=None)

        proxy_handler, redirect_handler = build_opener.call_args.args
        self.assertEqual(proxy_handler.proxies, {})
        self.assertIsInstance(redirect_handler, _NoRedirectHandler)
        self.assertIsNone(
            redirect_handler.redirect_request(
                request,
                None,
                307,
                "redirect",
                {},
                "https://attacker.example/collect",
            )
        )

    def test_config_repr_hides_key_and_rejects_remote_http(self) -> None:
        key = "unit-test-placeholder-secret"
        config = OpenAICompatibleConfig(
            base_url="https://provider.example/v1",
            model="test-model",
            api_key=key,
        )
        self.assertNotIn(key, repr(config))
        with self.assertRaisesRegex(ValueError, "requires HTTPS"):
            OpenAICompatibleConfig(
                base_url="http://provider.example/v1",
                model="test-model",
                api_key=key,
            )
        with self.assertRaisesRegex(ValueError, "control characters"):
            OpenAICompatibleConfig(
                base_url="https://provider.example/v1",
                model="test-model",
                api_key="placeholder\r\nX-Leak: yes",
            )

    def test_local_http_is_allowed_for_local_model_servers(self) -> None:
        config = OpenAICompatibleConfig(
            base_url="http://127.0.0.1:8000/v1",
            model="local-model",
            api_key="local-placeholder",
        )
        self.assertEqual(config.base_url, "http://127.0.0.1:8000/v1")

    def test_offline_transport_receives_valid_request(self) -> None:
        body = json.dumps({"choices": [{"message": {"content": "offline answer"}}]}).encode()
        transport = FakeTransport([TransportResponse(status=200, body=body)])
        config = OpenAICompatibleConfig(
            base_url="https://provider.example/v1",
            model="test-model",
            api_key="unit-test-placeholder",
            max_retries=0,
        )
        provider = OpenAICompatibleProvider(config, transport=transport)
        answer = provider.complete(
            [ChatMessage(role="user", content="hello")],
            max_tokens=12,
        )
        self.assertEqual(answer, "offline answer")
        request = transport.requests[0]
        self.assertEqual(request.full_url, "https://provider.example/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer unit-test-placeholder")
        payload = json.loads(request.data.decode())
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["messages"][0], {"role": "user", "content": "hello"})

    def test_error_does_not_expose_key_or_response_body(self) -> None:
        secret = "never-print-this-placeholder"
        response_body = f"server echoed {secret}".encode()
        transport = FakeTransport([TransportResponse(status=400, body=response_body)])
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                base_url="https://provider.example/v1",
                model="test-model",
                api_key=secret,
                max_retries=0,
            ),
            transport=transport,
        )
        with self.assertRaises(ProviderError) as caught:
            provider.complete([ChatMessage(role="user", content="hello")])
        message = str(caught.exception)
        self.assertNotIn(secret, message)
        self.assertNotIn("server echoed", message)
        self.assertIn("HTTP 400", message)

    def test_retryable_status_uses_bounded_offline_retry(self) -> None:
        success_body = json.dumps(
            {"choices": [{"message": {"content": "retried answer"}}]}
        ).encode()
        transport = FakeTransport(
            [
                TransportResponse(status=429, body=b"rate limited"),
                TransportResponse(status=200, body=success_body),
            ]
        )
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                base_url="https://provider.example/v1",
                model="test-model",
                api_key="unit-test-placeholder",
                max_retries=1,
            ),
            transport=transport,
        )
        with patch("tinygpt_forge.providers.openai_compatible.time.sleep") as sleep:
            answer = provider.complete([ChatMessage(role="user", content="hello")])
        self.assertEqual(answer, "retried answer")
        self.assertEqual(len(transport.requests), 2)
        sleep.assert_called_once_with(0.25)

    def test_request_and_response_size_limits_are_enforced(self) -> None:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                base_url="https://provider.example/v1",
                model="test-model",
                api_key="unit-test-placeholder",
                max_retries=0,
            ),
            transport=FakeTransport(
                [TransportResponse(status=200, body=b"x" * (MAX_PROVIDER_RESPONSE_BYTES + 1))]
            ),
        )
        with self.assertRaisesRegex(ValueError, "total message content"):
            provider.complete([ChatMessage(role="user", content="x" * 100_000) for _ in range(11)])
        with self.assertRaisesRegex(ProviderError, "2 MiB"):
            provider.complete([ChatMessage(role="user", content="hello")])


if __name__ == "__main__":
    unittest.main()
