# Security review

Review date: 2026-08-10

## Executive summary

No known critical vulnerability remains in the current local-first scope. This review found and
fixed one high-severity credential-boundary issue and one medium-severity metadata parser issue.
Untrusted tensor resource exhaustion remains an explicit local denial-of-service boundary; the
project is not a sandbox or multi-tenant serving platform.

## High severity

### SEC-001 — Bearer token could cross an HTTP redirect (fixed)

Impact: a configured endpoint returning a redirect could cause the standard HTTP stack to submit
the API key to a different origin.

The transport now installs a handler that refuses every redirect before a replacement request is
created ([implementation](../src/tinygpt_forge/providers/openai_compatible.py#L33),
[opener boundary](../src/tinygpt_forge/providers/openai_compatible.py#L72)). A 3xx therefore becomes
a sanitized provider error. The same configuration validation rejects ASCII control characters
before constructing the Authorization header
([validation](../src/tinygpt_forge/providers/openai_compatible.py#L126)). Offline tests assert the
no-redirect handler and never contact a real service.

## Medium severity

### SEC-002 — Unbounded or ambiguous JSON/TOML metadata (fixed)

Checkpoint, tokenizer, and configuration readers previously loaded entire metadata files and used
the permissive Python JSON defaults. An adversarial file could consume excessive parser memory;
duplicate keys or `NaN`/`Infinity` could be interpreted inconsistently by other tools.

The shared reader now consumes at most limit+1 bytes, defaults to a 2 MiB ceiling, rejects duplicate
keys recursively, rejects non-finite constants, requires strict UTF-8, and requires an object root
([bounded read](../src/tinygpt_forge/serialization.py#L12),
[strict JSON](../src/tinygpt_forge/serialization.py#L43)). It is used by checkpoint manifests,
Tokenizer JSON, and TOML configuration. Tests cover oversize, duplicate, nonfinite, non-object, and
invalid-UTF-8 inputs.

### SEC-003 — Untrusted tensor shapes can exhaust CPU/GPU memory (accepted boundary)

Safetensors prevents pickle code execution, but a valid manifest and tensor file can still describe
a model too large for the machine. `load_weights` validates the bounded manifest and configuration,
then allocates the model before strict tensor loading
([allocation path](../src/tinygpt_forge/checkpoint.py#L87)). A universal dimension cap would reject
legitimate research models and would not replace OS/GPU quotas.

Only load checkpoints from a trusted source, inspect size/configuration first, and use an isolated
process with filesystem, network, CPU, GPU, and memory limits for untrusted artifacts. This project
does not claim sandboxed model loading.

## Low severity and operational risks

### SEC-004 — SHA-256 provides integrity, not publisher identity

The manifest digest detects byte changes and mixed bundles, but an attacker controlling both tensor
and manifest files can recompute it. Releases should use GitHub's authenticated history/release
artifacts; signed release attestations remain future work.

### SEC-005 — Dependency installation is not bit-for-bit locked

The library declares compatible runtime ranges while developer tools are pinned. This is appropriate
for a reusable Python package but means a future resolver can choose different compatible wheels.
The CI definition pins and separates the minimum declared combination (Python 3.10, PyTorch 2.3.0)
from the current combination (Python 3.14, PyTorch 2.13.0). These jobs remain unverified until they
run on GitHub Actions. Users needing a frozen experiment should capture the complete resolved
environment and install only from trusted indexes.

## Informational boundaries

- External endpoints receive submitted prompts and the configured authorization header and may
  charge the account. The explicit cost flag is acknowledgement, not a spending limit.
- API keys necessarily exist in process memory while a request is built. Hiding dataclass `repr`
  prevents accidental logs, not process compromise.
- The optional client is synchronous and single-user. It does not implement authentication,
  admission control, quotas, or tenant isolation and must not be presented as a production server.
- Local training/inference and all CI tests remain independent of external API credentials.
