# Optional external API boundary

TinyGPT Forge is local-first. Model training, checkpoint loading, generation, tests, and
benchmarks do not require an API key or network connection.

The optional `external-chat` command is a small OpenAI-compatible client, not a production
server. It exists to demonstrate a provider boundary and interoperability with explicitly chosen
services, including a local compatible server.

## Configuration

Copy `.env.example` to a local `.env` only if your own shell or secret manager loads it; the
project does not silently parse arbitrary files. `.env` is ignored by Git. The required process
environment variables are:

- `TINYGPT_API_KEY`
- `TINYGPT_BASE_URL`
- `TINYGPT_MODEL`

Timeout, retry count, and proxy are optional namespaced variables. Remote endpoints require
HTTPS; plain HTTP is accepted only for localhost. URLs with embedded usernames/passwords, query
strings, or fragments are rejected.

## Explicit cost acknowledgement

An external call may transmit prompt content to another operator and may cost money. The CLI
refuses to send a request unless the caller adds:

```text
--yes-i-understand-this-may-cost-money
```

No CI test makes a real request. Tests inject an offline transport and verify request structure,
retry/error behavior, and that API keys or provider response bodies do not appear in exceptions.

## Security boundary

- Keys are read from explicit environment variables and excluded from configuration `repr`.
- Request timeouts and retry counts are bounded.
- Per-message length, one-million-character aggregate input, requested output length, and
  temperature are validated.
- The standard transport reads at most 2 MiB plus one sentinel byte; larger responses are rejected.
- Provider errors expose a status code but not response bodies, headers, or credentials.
- The standard transport supports an explicit proxy but does not log it.

This client is experimental until it has interoperability tests against named local servers. It
does not claim streaming, tool calling, multi-provider normalization, or production hardening.
