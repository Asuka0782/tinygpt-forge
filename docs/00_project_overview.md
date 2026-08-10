# Project overview and implementation contract

## The first correctness contract

The baseline maps integer token IDs to next-token logits:

```text
input_ids [batch, time]
  -> token embeddings [batch, time, d_model]
  -> N x pre-norm decoder block
  -> RMSNorm
  -> LM head
  -> logits [batch, time, vocab]
```

Each decoder block uses:

```text
x = x + Attention(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

For `n_heads = Hq`, `n_kv_heads = Hkv`, and `head_dim = d_model / Hq`:

- query shape: `[B, Hq, T, Dh]`
- key/value shape before grouping: `[B, Hkv, T, Dh]`
- key/value shape after grouped-query expansion: `[B, Hq, T, Dh]`
- attention scores in the manual backend: `[B, Hq, T, T]`

MHA is the special case `Hq = Hkv`; GQA requires `Hq` to be divisible by `Hkv`.

## Loss semantics

There are two intentionally separate APIs:

- `aligned_next_token_cross_entropy(logits, next_token_ids)` expects an already shifted
  `(x, y)` batch and scores every logit position.
- `shifted_next_token_cross_entropy(logits, token_ids)` accepts one complete token sequence,
  drops the final logit and the first token, then computes the same next-token objective.

Keeping both names explicit prevents a common silent bug: shifting a batch twice or not at all.

## Evidence gates

Before adding training or performance claims, the baseline must pass:

1. config and shape invariants;
2. RoPE norm and relative-position checks;
3. manual/SDPA forward and backward equivalence at dropout zero;
4. causal output and future-gradient checks;
5. MHA and GQA shape checks;
6. hand-computed loss equivalence.

This tiny-case oracle is the rollback point for every later optimization.

## KV cache contract

Each layer caches the unexpanded key/value tensors with shape `[B, Hkv, S, Dh]`. Keeping
`Hkv` rather than repeating to `Hq` preserves GQA's cache-memory benefit. A decode step at
absolute position `S` may attend to key positions `0..S`; the implementation therefore uses
an explicit non-square causal mask for cached SDPA instead of assuming that `is_causal=True`
has the desired alignment.

`StaticKVCache` preallocates `[B, Hkv, capacity, Dh]` key/value storage for every layer and
writes each decode position in place. A shared logical length advances only after all layers
complete, preventing one layer from observing a different prefix length than another.

## Data and checkpoint boundaries

The built-in character tokenizer is fitted on training text only. Token IDs are split into
contiguous train/validation/test segments, and `NextTokenBatcher` owns an independent RNG
state so a resumed run can reproduce its next sampled window. Model weights are stored in
safetensors with a separate JSON config and SHA-256 digest. Optimizer and trainer-state
resume are a separate contract and are not implied by a weights-only checkpoint.
