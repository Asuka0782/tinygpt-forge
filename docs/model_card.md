# Model and experiment card

## Model family

TinyGPT Forge implements a configurable decoder-only causal language model with token embeddings,
pre-normalized residual blocks, RoPE, grouped-query attention, SwiGLU, final RMSNorm, and a tied
language-model head.

The default CPU configuration has 106,816 parameters after the character vocabulary is resolved.
GPU benchmark configurations with 3,148,032 and 24,650,240 parameters use random weights because
the benchmark targets execution mechanisms rather than language quality.

## Training objective

For aligned input/target windows, position `t` predicts the next token exactly once. The package
also exposes an explicit shifted-sequence loss. Unit tests compare both conventions with manual
flattened cross-entropy to prevent double-shift and missing-shift bugs.

## Evidence

- Manual/SDPA forward and backward equivalence at dropout zero.
- Future tokens do not change past outputs; future embedding gradients are zero.
- RoPE preserves pairwise norm and global-shift attention dot products within float64 tolerance.
- Dynamic/Static cache incremental logits and greedy generation match full recomputation.
- Safetensors weight reload is bitwise exact.
- Four uninterrupted training steps match two steps plus resume bitwise, including dropout and
  AdamW continuation state.
- A real five-step RTX 5060 BF16 smoke completed fused-AdamW training, best-checkpoint reload, and
  CUDA generation; its [full artifact](../results/training/rtx5060_bf16_smoke.json) retains the
  repetitive output and does not support a language-quality claim.

See `docs/10_benchmarks.md` and `results/benchmarks/` for performance conditions and raw samples.

## Intended use

- Learning GPT mathematics and tensor shapes.
- Studying the boundary between semantic equivalence and systems performance.
- Reproducing small CPU/CUDA experiments before scaling them.

## Out-of-scope use

The bundled model/corpus must not be presented as a useful assistant, foundation model, safety
model, or benchmark-quality language model. No model weights are distributed as a release artifact.

## Known limitations

- Character tokenization is inefficient for realistic language modeling.
- The fixture has no meaningful generalization or safety coverage.
- Results are sensitive to model, batch, context, dtype, backend, OS, power, and temperature.
- Current evidence does not cover distributed training or production serving.
