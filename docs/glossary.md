# Glossary

**Autocast** — PyTorch context that selects lower precision for eligible operations while retaining
higher precision where needed.

**Autograd** — automatic differentiation system that records tensor operations and computes
gradients by reverse-mode chain rule.

**Batch (`B`)** — independent sequences processed together in one model call.

**BF16** — 16-bit floating point with FP32-like exponent width and reduced mantissa precision.

**Causal mask** — rule preventing a token from attending to future positions.

**Decode** — autoregressive stage that produces new tokens after prompt prefill.

**Dynamic KV Cache** — cache that grows by concatenating new key/value tensors.

**FlashAttention** — IO-aware exact attention family that reduces high-bandwidth-memory traffic via
tiling/fusion; also the name of a specific optional implementation package.

**FP16 / FP32** — IEEE half/single-precision floating-point formats.

**Gradient accumulation** — multiple backward microbatches before one optimizer step.

**GQA (Grouped-Query Attention)** — several query heads share one key/value head.

**Head dimension (`Dh`)** — per-query-head width, `d_model/n_heads`.

**KV Cache** — stored attention keys/values reused during autoregressive decode.

**LM head** — projection from final hidden states to vocabulary logits.

**Logit** — unnormalized score before softmax.

**LoRA** — low-rank trainable update attached to a frozen weight matrix.

**MHA (Multi-Head Attention)** — equal numbers of query and key/value heads.

**MQA (Multi-Query Attention)** — all query heads share one key/value head.

**OOV (Out of Vocabulary)** — input unit absent from the fitted vocabulary.

**Perplexity** — exponentiated mean token negative log-likelihood under a fixed evaluation convention.

**Prefill** — processing all prompt tokens and constructing initial cache state.

**Pre-norm** — normalization applied before attention/feed-forward sublayers.

**QLoRA** — LoRA training with a quantized frozen base model and higher-precision adapter compute.

**RMSNorm** — root-mean-square normalization without mean subtraction.

**RoPE** — rotary position embedding applied to query/key channel pairs.

**SDPA** — scaled dot-product attention API that dispatches among available PyTorch backends.

**Static KV Cache** — fixed-capacity cache updated in place.

**SwiGLU** — gated feed-forward activation `silu(gate(x))*up(x)` followed by down projection.

**Teacher forcing** — training with real previous tokens as context while scoring all next tokens in
parallel.

**Tensor Core** — GPU matrix-multiply hardware optimized for supported lower-precision formats.

**TF32** — NVIDIA/PyTorch matmul mode using FP32 range with reduced multiplication mantissa.

**Time to first token (TTFT)** — latency from request start through prefill and first output token.

**Token throughput** — tokens processed/generated per second; meaningful only with tokenizer, stage,
batch, and measurement method specified.

