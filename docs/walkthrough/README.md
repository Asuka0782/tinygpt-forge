# Maintained line-by-line walkthrough

These notes bind explanations to physical source ranges. A range may group a docstring, import
block, or one indivisible tensor expression; executable statements are kept in small groups so the
syntax, data flow, shapes, systems effect, alternative, and boundary can be explained together.

The documentation test checks that declared ranges start at their linked line and, after this tour
is complete, that every physical line in `src/tinygpt_forge/` is covered. The shorter
[`docs/11_source_code_walkthrough.md`](../11_source_code_walkthrough.md) remains the execution-path
overview; this directory is the detailed companion.

Current sections:

1. [Components, RoPE, loss, Static Cache, and generation](01_components_cache_generation.md)
2. [Attention and the complete decoder model](02_attention_and_gpt.md)
3. [Configuration, data splitting, tokenization, and TOML compatibility](03_config_data_tokenizer.md)
4. [Safe weights and complete training checkpoints](04_checkpoint.md)
5. [Training configuration, evaluation, resume, and experiment manifests](05_training.md)
6. [Optional provider boundary and OpenAI-compatible client](06_providers.md)
7. [Correctness-gated benchmark harness and evidence schema](07_benchmark.md)
8. [CLI dispatch and package entry points](08_cli_and_entrypoints.md)
9. [Bounded strict metadata parsing](09_serialization.md)
