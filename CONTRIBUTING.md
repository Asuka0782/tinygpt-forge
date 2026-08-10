# Contributing

TinyGPT Forge accepts changes only when their semantics and evidence are reviewable. A feature is
not complete merely because one script ran faster once.

## Local setup

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
python -m unittest discover -s tests -v
tinygpt smoke --config configs/tiny_cpu.toml
```

CPU tests must remain runnable without CUDA, an API key, or network access. GPU-only behavior must
have an explicit compatibility check and a safe fallback where applicable.

## Change contract

For model, training, or inference changes, state:

1. the input/output tensor shapes and mathematical semantics;
2. the baseline under the same model, data, dtype, shape, and budget;
3. correctness checks and numeric tolerances;
4. benchmark warm-up, synchronization, repeats, raw samples, hardware, and software;
5. cases where the change is slower, less accurate, unsupported, or unverified.

Do not promote an experimental feature to the README until its tests and evidence exist in the
repository. Preserve failed or negative benchmark results when they materially limit a claim.

## Source and license hygiene

- Do not paste source of unknown origin.
- Record papers, repositories, versions, and licenses in `docs/source_and_license_matrix.md`.
- Preserve third-party copyright, LICENSE, and NOTICE requirements for any direct adaptation.
- Do not describe an adaptation as original implementation.
- Do not add datasets or model weights until their redistribution terms are documented.

## Secrets and private data

Never commit `.env`, API keys, tokens, cookies, local account paths, private prompts, model caches,
or unreviewed checkpoints. External provider tests must inject an offline transport. Sanitized
exceptions must not contain request headers or response bodies.

## Pull request checklist

- [ ] Scope is small enough to review.
- [ ] Tests cover the new success and failure paths.
- [ ] Ruff lint and format checks pass.
- [ ] CPU tests pass without external services.
- [ ] Claims are supported by source or experiment artifacts.
- [ ] New dependency and license costs are documented.
- [ ] Documentation and status labels match actual support.
- [ ] No generated caches, large files, credentials, or private paths are included.

