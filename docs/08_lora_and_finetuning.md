# LoRA and fine-tuning: design and admission gates

## Status

LoRA/QLoRA are **not stable TinyGPT Forge features yet**. This chapter explains the mechanism and
the tests required before implementation is promoted. The source learning experiments established
the engineering checks, but their tiny dataset did not establish generalization.

## Full fine-tuning

Full fine-tuning updates every parameter. For a model with `P` parameters, training also holds
gradients and optimizer moments; AdamW state can dominate weight memory. It offers maximum
adaptation capacity but is expensive to store and train separately for many tasks.

## LoRA

For frozen weight `W in R^{d_out × d_in}`, LoRA learns a low-rank update:

\[
W'=W+\frac{\alpha}{r}BA,
\]

where `A in R^{r × d_in}`, `B in R^{d_out × r}`, and `r` is small. Trainable parameter count is
`r(d_in+d_out)` rather than `d_in d_out`.

A common initialization samples `A` and initializes `B=0`. The initial update is exactly zero, so
the adapted model starts as a no-op. On the first backward pass, `B` can receive nonzero gradient
while `A` gradient is initially zero because it is multiplied through zero `B`. This is expected,
not a dead adapter.

## Target modules

Applying LoRA to query/value projections is common but not universally optimal. Possible targets
include Q/K/V/O attention projections and feed-forward projections. A feature must report exact
target names, ranks, scaling, dropout, trainable parameter count, and matched training budget.

## SFT labels and masking

Supervised fine-tuning often concatenates instruction/prompt and completion. If the objective is to
learn only the completion, prompt and padding labels must be `ignore_index` so they do not contribute
to cross-entropy. A mask test should count supervised tokens and verify manual/internal loss.

## QLoRA

QLoRA keeps the frozen base model in a low-bit representation while training LoRA adapters, often
with higher-precision computation and optimizer state. It reduces weight memory but adds quantize/
dequantize kernels, backend constraints, and numeric behavior.

“Loads in 4-bit” is not enough. Admission requires:

- exact quantization format and compute dtype;
- Windows/Linux, Python, CUDA, and GPU compatibility matrix;
- peak memory including temporary buffers;
- adapter gradient and no-op tests;
- save/reload and merge policy;
- quality comparison under matched data/steps/seeds;
- clear distinction between training and inference quantization.

## Required tests before promotion

1. Base/adapter no-op equivalence at initialization.
2. Trainable-parameter allowlist; base gradients absent.
3. Expected first-step A/B gradient behavior.
4. Label-mask and shifted-loss correctness.
5. Adapter save/reload exactness within dtype tolerance.
6. Disable-adapter returns base output.
7. Merge/unmerge equivalence and documented irreversible cases.
8. Resume includes adapter, optimizer, RNG, scheduler, and data cursor.
9. Multiple seeds or appropriately narrow claims.

## Evidence boundaries from tiny SFT

A very small dataset can show that loss decreases, adapters update, and serialization works. It
cannot show broad instruction following, safety, factuality, or task generalization. Related unseen
prompts that fail are valuable evidence and must remain in the experiment report.

## Planned integration

The preferred implementation is an optional PEFT bridge rather than a second home-grown adapter
ecosystem. Core TinyGPT training remains independent of Transformers/PEFT. Offline tests will use a
tiny local model and never download weights or call an API in CI.
