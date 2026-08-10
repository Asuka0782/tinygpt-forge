# Causal self-attention from scratch

## Shapes first

Let input `X` have shape `[B,T,C]`:

- `B`: batch size;
- `T`: sequence length;
- `C`: model width (`d_model`);
- `Hq`: query heads;
- `Hkv`: key/value heads;
- `Dh = C/Hq`: head dimension.

Projection and reshape produce:

```text
Q: [B,Hq,T,Dh]
K: [B,Hkv,T,Dh]
V: [B,Hkv,T,Dh]
```

For GQA, `Hq/Hkv` query heads share each K/V head. The readable implementation repeats K/V only for
the attention computation; caches retain `[B,Hkv,S,Dh]` storage.

## Score, mask, softmax, value mix

For each batch and head:

\[
S_{ij}=\frac{q_i^\top k_j}{\sqrt{D_h}}.
\]

Training without cache uses the lower-triangular rule `j <= i`. After masking:

\[
A_{ij}=\frac{\exp S_{ij}}{\sum_{k\le i}\exp S_{ik}},\qquad
y_i=\sum_{j\le i}A_{ij}v_j.
\]

The manual path materializes scores and probabilities `[B,Hq,T,T]`. This is ideal for inspection
but uses quadratic memory.

## The cached non-square mask

If the cache contains `S` old tokens and the current query has length `Q`, score shape is
`[B,Hq,Q,S+Q]`. Query row `i` represents absolute position `S+i`, so it may read key indices
`j <= S+i`.

This alignment is not the same as blindly applying a top-left triangular `Q × (S+Q)` mask. TinyGPT
Forge constructs the absolute-position boolean mask explicitly when query/key lengths differ.

## Manual versus SDPA

The manual backend is the correctness oracle:

1. project and apply RoPE;
2. expand GQA K/V for query heads;
3. compute scaled scores;
4. apply causal mask;
5. softmax and dropout;
6. multiply values, merge heads, output projection.

The SDPA backend calls `torch.nn.functional.scaled_dot_product_attention`. PyTorch may dispatch to
math, memory-efficient, FlashAttention, or another compiled backend depending on build, device,
dtype, shape, and masks. A configuration flag does not prove which kernel ran.

At dropout zero, both backends are tested in float64 for:

- output equivalence;
- input-gradient equivalence;
- parameter-gradient equivalence.

At lower precision, tolerances must reflect dtype and accumulation behavior.

## Causality tests

Checking only the mask tensor is insufficient. The suite verifies causality at three levels:

- upper-triangular manual probabilities are exactly zero;
- replacing suffix inputs does not change prefix outputs;
- a loss at position `t` has zero gradient with respect to future input embeddings.

The gradient test catches accidental future paths outside the obvious score mask.

## Complexity and GPU meaning

Ignoring projections, score/value work is `O(B Hq T² Dh)` and attention probabilities occupy
`O(B Hq T²)`. Projection/MLP work is often dominant for small `T`; attention becomes more important
as context grows.

SDPA can fuse operations and avoid writing the complete probability matrix to high-bandwidth memory.
It may be slower for tiny shapes because kernel launch and dispatch overhead dominate. This is why
benchmarks sweep shape rather than advertise one ratio.

## Autograd

Gradients flow through softmax to Q/K and directly through the weighted sum to V. Saturated softmax
can make most score gradients tiny. The scale factor helps maintain a useful score range. Dropout on
attention probabilities introduces stochastic masks and therefore requires RNG state for exact
training resume.

## Code map

- `model/attention.py::CausalSelfAttention` owns projection, mask, manual/SDPA selection, and cache
  integration.
- `model/rope.py::RotaryEmbedding` rotates Q/K.
- `cache.py::StaticKVCache` stores unexpanded K/V.
- `tests/test_attention.py` is the small mathematical oracle.

