## What changed

Describe the smallest useful change and the problem it solves.

## Validation

- [ ] Relevant unit or integration tests pass.
- [ ] `ruff check .` and `ruff format --check .` pass.
- [ ] `mypy src` passes when source types changed.
- [ ] Correctness was checked before performance when this is an optimization.
- [ ] Benchmarks include raw repeats, warm-up, synchronization, shape, dtype, and device.
- [ ] Documentation and feature-status claims match what was actually validated.

## Safety and provenance

- [ ] No credentials, private paths, caches, checkpoints, or unrelated generated files are included.
- [ ] New external code or data has a documented source and compatible license.
- [ ] No third-party attribution or license text was removed.

## Limits

State what this pull request does not validate, especially across devices, Python versions, or model scales.
