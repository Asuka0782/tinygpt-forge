# Line-by-line: optional provider boundary

TinyGPT Forge is local-first: training and generation never import this package. This module exists
only to demonstrate how an external OpenAI-compatible service can be placed behind a small,
testable boundary without putting credentials into source code or CI.

## `providers/base.py`

| Source lines | Explanation |
|---|---|
| [L1–L8](../../src/tinygpt_forge/providers/base.py#L1) | The module docstring, postponed annotations, `Sequence`, frozen dataclasses, and `Protocol` support a dependency-free interface. `Sequence` accepts tuples/lists without promising mutation. |
| [L9–L12](../../src/tinygpt_forge/providers/base.py#L9) | `ProviderError` is the public, sanitized failure type. Provider bodies and authorization headers must stay in the exception cause boundary, never in this message. |
| [L13–L19](../../src/tinygpt_forge/providers/base.py#L13) | A frozen `ChatMessage` carries only role and text. Immutability prevents the request payload from changing between validation and transmission. |
| [L20–L27](../../src/tinygpt_forge/providers/base.py#L20) | Construction accepts only the three chat roles, rejects empty content, and caps each message at 100,000 characters. This is input validation and memory-risk reduction, not prompt-injection protection. |
| [L28–L40](../../src/tinygpt_forge/providers/base.py#L28) | `ChatProvider` is structural typing: any object with this synchronous `complete` signature qualifies without inheritance. The keyword-only generation controls make call sites legible. A production streaming API would require a separate async/iterator contract rather than weakening this one. |

## `providers/openai_compatible.py`

| Source lines | Explanation |
|---|---|
| [L1–L18](../../src/tinygpt_forge/providers/openai_compatible.py#L1) | Standard-library JSON, environment, retry, HTTP, typing, and URL tools keep the optional client dependency-free. Response bytes and aggregate message characters have explicit upper bounds. |
| [L19–L27](../../src/tinygpt_forge/providers/openai_compatible.py#L19) | Frozen `TransportResponse` is the narrow result of HTTP transport: status plus bounded raw bytes. Parsing remains a provider responsibility, so tests can inject responses without a network. |
| [L28–L46](../../src/tinygpt_forge/providers/openai_compatible.py#L28) | The internal connection exception distinguishes retryable I/O. `_NoRedirectHandler` returns no replacement request for every 3xx, so `urllib` turns the redirect into an HTTP error instead of forwarding the bearer header to another origin. `Any` is limited to the untyped standard-library override parameters. |
| [L47–L59](../../src/tinygpt_forge/providers/openai_compatible.py#L47) | `HTTPTransport` is a structural interface receiving the already-built request, finite timeout, and explicit proxy. Tests replace this boundary and never call a paid endpoint. |
| [L60–L77](../../src/tinygpt_forge/providers/openai_compatible.py#L60) | `UrllibTransport` turns `None` into an empty proxy map, disabling implicit system-proxy discovery; otherwise the validated proxy handles both schemes. The opener always installs the proxy and no-redirect handlers, making the credential destination deterministic. |
| [L78–L90](../../src/tinygpt_forge/providers/openai_compatible.py#L78) | A finite-timeout request reads at most limit+1 bytes so the caller can detect overflow. HTTP errors—including redirects—retain bounded status/body; URL, timeout, and OS failures become a credential-free retryable error. |
| [L91–L104](../../src/tinygpt_forge/providers/openai_compatible.py#L91) | URL validation requires an absolute HTTP(S) host, rejects embedded username/password and query/fragment ambiguity, permits cleartext HTTP only for loopback hosts, and removes a trailing slash. This is a client boundary, not a complete SSRF defense for a future multi-user server. |
| [L105–L117](../../src/tinygpt_forge/providers/openai_compatible.py#L105) | Frozen endpoint configuration hides `api_key` from dataclass `repr`, while retaining base URL, model, timeout, bounded retries, and optional proxy. Hiding repr reduces accidental logs; it cannot protect a compromised process. |
| [L118–L133](../../src/tinygpt_forge/providers/openai_compatible.py#L118) | Post-init canonicalizes the base URL, bounds model/key length, rejects ASCII controls that could corrupt or expose an Authorization header, and bounds timeout/retry count. Frozen instances mutate only validated canonical fields through `object.__setattr__`. |
| [L134–L143](../../src/tinygpt_forge/providers/openai_compatible.py#L134) | A configured proxy goes through the same credential/scheme/query/local-HTTP checks. No proxy means the transport installs an empty map rather than consulting unrelated environment variables. |
| [L144–L159](../../src/tinygpt_forge/providers/openai_compatible.py#L144) | `from_environment` reads only the supplied mapping or process environment and lists missing variable names, never values. Passing a mapping makes configuration tests hermetic. |
| [L160–L167](../../src/tinygpt_forge/providers/openai_compatible.py#L160) | Required namespaced values and optional timeout/retry/proxy strings become the validated dataclass. `.env` loading stays outside the library so callers control secret ingestion. |
| [L168–L181](../../src/tinygpt_forge/providers/openai_compatible.py#L168) | The provider owns immutable config and either an injected fake/custom transport or the standard implementation. Constructor injection is why CI can verify payload and failures without a key or network. |
| [L182–L201](../../src/tinygpt_forge/providers/openai_compatible.py#L182) | `complete` validates nonempty/bounded message count, aggregate characters, output-token range, and temperature before serialization. These controls bound obvious resource abuse but do not estimate provider-specific tokenizer counts. |
| [L202–L212](../../src/tinygpt_forge/providers/openai_compatible.py#L202) | Messages become conventional chat-completions JSON and UTF-8 bytes. `ensure_ascii=False` preserves readable non-ASCII payloads while the transport encoding remains UTF-8. |
| [L213–L222](../../src/tinygpt_forge/providers/openai_compatible.py#L213) | The request appends `/chat/completions`, uses POST, and adds bearer authorization, JSON content type, and a versioned user agent. The request object contains the key, so callers must never log or snapshot it. |
| [L223–L235](../../src/tinygpt_forge/providers/openai_compatible.py#L223) | At most `max_retries+1` attempts call the transport with explicit timeout/proxy. A final connection failure becomes public `ProviderError`; earlier failures proceed to bounded backoff without echoing URL internals or credentials. |
| [L236–L245](../../src/tinygpt_forge/providers/openai_compatible.py#L236) | Every body is size-checked before parsing. 2xx parses content; 408/409/429/5xx are retryable, other statuses—including 3xx—fail immediately, and exponential waits cap at two seconds. This small client does not interpret `Retry-After` or add jitter. |
| [L246–L258](../../src/tinygpt_forge/providers/openai_compatible.py#L246) | The terminal raise is a defensive fallback. Parsing demands UTF-8 JSON at `choices[0].message.content`, converts structural/decoding failures into one sanitized error, and rejects empty/non-string output. Provider-specific extensions are ignored. |

## `providers/__init__.py`

| Source lines | Explanation |
|---|---|
| [L1–L8](../../src/tinygpt_forge/providers/__init__.py#L1) | The package docstring states its optional status and imports only the stable interface/config/client names from implementation modules. |
| [L9–L15](../../src/tinygpt_forge/providers/__init__.py#L9) | `__all__` defines the supported public surface for wildcard imports and documentation tools; internal transport details remain available for tests but are not advertised as stable API. |

## Data and systems path

`ChatMessage[] → validated JSON bytes → HTTPTransport → bounded status/body → validated text`.
Unlike local generation there are no model tensors or autograd graph here. Latency and cost belong
to the external service; the project records this feature as experimental and keeps all core tests
offline.
