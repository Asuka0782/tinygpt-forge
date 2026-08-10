# Autoregressive inference and KV Cache

## Prefill and decode

Given prompt length `P`, prefill processes all prompt tokens and produces logits plus per-layer
keys/values. Decode then produces one token at a time:

```text
prefill: [B,P] -> logits [B,P,V] + cache length P
decode:  [B,1] + cache length S -> logits [B,1,V] + cache length S+1
```

Time to first token includes prompt processing and first sampling. Steady-state decode latency
measures later single-token steps. Combining them into one number can hide which stage improved.

## Full recomputation

The simplest generator appends a token and feeds the entire growing sequence back to the model.
For each step, all old Q/K/V projections, block computations, and attention rows are repeated. It is
a clear correctness oracle and can still be surprisingly fast for tiny shapes because large
full-sequence kernels use the GPU well.

## Dynamic Cache

Each layer receives old `[B,Hkv,S,Dh]` K/V and current `[B,Hkv,Q,Dh]` K/V, concatenates them, and
returns the longer tensors. It is simple and supports arbitrary growth up to context length.

Cost: every step allocates/copies the growing cache. For tiny models this overhead can dominate the
saved arithmetic.

## Static Cache

`StaticKVCache` preallocates:

```text
K,V per layer: [B,Hkv,capacity,Dh]
```

Each layer writes the current range in place. A shared logical length advances only after every layer
has written, so layers cannot observe inconsistent prefix lengths. `rewind()` enables steady-state
benchmark reuse without reallocating or exposing discarded suffix values.

Static allocation does not guarantee lower peak memory in end-to-end measurement: the explicit
capacity is live throughout generation. Its benefit is stable addresses and avoided concatenation,
which may help compilation or larger workloads.

## Cached causal mask

For past length `S` and current query row `i`, allowed key index is:

\[
j\le S+i.
\]

The project uses an explicit boolean mask for non-square cached attention. For square training
attention with no past, it uses `is_causal=True`. This distinction avoids the common top-left versus
bottom-right alignment mistake.

## Sampling semantics

Greedy sampling (`temperature=0`) selects argmax. For temperature `tau>0`:

\[
p_i=\operatorname{softmax}(z_i/\tau).
\]

Top-k masks logits below the kth largest value before softmax. A device-local generator with a fixed
seed controls multinomial sampling. Full/Dynamic/Static paths call the same sampler so differences in
generated IDs indicate model/cache divergence rather than different sampling code.

## Complexity

At one decode step with prefix `S`, full recomputation processes `S+1` token states through every
block. Cached decode processes one new token through projections/MLP and attends one query to
`S+1` keys. Algorithmic work is much smaller, but wall-clock speed depends on parallel utilization,
memory traffic, allocations, masks, and kernel launch overhead.

Cache bytes for FP32 are:

\[
2LBH_{kv}SD_h\times4.
\]

GQA/MQA reduce `Hkv`; they do not reduce the number of query heads computing output.

## Measured crossover

On the tested RTX 5060 laptop:

- batch-one small/medium/24.65M models did not gain latency from cache;
- 24.65M parameters, batch eight, prompt 128 crossed over;
- Dynamic/Static end-to-end speedups were about 1.33×/1.32×;
- Static steady-state single-token speedup was about 1.39×.

These are local shape boundaries. See `10_benchmarks.md` for raw-artifact names, memory results, and
method limitations.

## Correctness gates

- full versus chunked incremental logits within tolerance;
- full/Dynamic/Static greedy IDs exactly equal;
- cache shape retains `Hkv`, not repeated `Hq`;
- capacity, device, dtype, batch, layer, and logical-position validation;
- prompt plus generation length cannot exceed configured context.

