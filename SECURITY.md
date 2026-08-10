# Security policy

## Reporting

After the public repository is created, report vulnerabilities with GitHub private security
advisories rather than a public issue. The project does not yet publish a stable supported release.

## Checkpoints and data

TinyGPT Forge writes model and optimizer tensors with safetensors plus validated JSON manifests and
SHA-256 digests. It does not use `torch.load` for public checkpoints. JSON/TOML metadata is limited
to 2 MiB and rejects duplicate object keys, invalid UTF-8, and non-finite JSON constants.
Safetensors prevents pickle code execution, but it does not make an untrusted model or dataset
benign: validate size, origin, license, and expected architecture before loading.

Large or adversarial tensor shapes can still cause memory exhaustion. Run untrusted artifacts in an
isolated environment with filesystem, network, CPU, GPU, and memory limits.

## API keys and external endpoints

- Local training and inference do not need a key.
- Keys are accepted through namespaced environment variables and excluded from configuration repr.
- `.env` and `.env.*` are ignored; `.env.example` contains placeholders only.
- Remote HTTP is rejected; plain HTTP is allowed only for localhost-compatible servers.
- HTTP redirects are rejected so bearer credentials cannot cross the configured endpoint boundary.
- API keys containing ASCII control characters are rejected before HTTP header construction.
- System proxy discovery is disabled; only the explicitly configured proxy is used.
- External requests require an explicit cost acknowledgement flag.
- Provider exceptions omit response bodies, headers, prompts, and credentials.

Do not treat a custom `base_url` as trusted. An endpoint receives submitted prompts and authorization
headers and may charge the configured account.

## Denial-of-service boundaries

The optional client bounds message count, per-message and aggregate content, provider response size,
requested output length, timeout, and retry count. TinyGPT configurations make context length
explicit but do not impose a universal maximum model/tensor size, because valid research workloads
vary widely. These are basic local guardrails, not a production multi-tenant security boundary.

See the [security review](docs/security_review.md) for fixed findings and accepted residual risks.
