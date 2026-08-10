# Performance optimization without wishful thinking

## Start with a bottleneck hypothesis

An optimization should name the constrained resource:

- compute throughput;
- high-bandwidth-memory traffic;
- activation/cache capacity;
- Python/kernel-launch overhead;
- communication;
- data input or serialization.

Reducing FLOPs helps only if compute is limiting. KV Cache is the project's concrete example: it
reduces repeated arithmetic but loses at batch one because tiny decode kernels underutilize the GPU.

## Manual attention and SDPA

Manual attention materializes `[B,H,T,T]` probabilities. SDPA can fuse score, mask, softmax, dropout,
and value mixing or tile them without writing the full matrix.

Backend selection is runtime-dependent. TinyGPT Forge records observed ATen operator names. The
tested wheel dispatches `aten::_scaled_dot_product_efficient_attention`; FlashAttention is not
compiled. A backend flag alone is not accepted as evidence.

## FlashAttention

FlashAttention reorganizes exact attention around on-chip tiling and reduced HBM reads/writes. It
does not change the mathematical softmax objective apart from floating-point ordering. Practical
support depends on OS, CUDA, GPU architecture, dtype, head dimension, build toolchain, and package
wheel availability.

Policy:

- optional dependency only;
- import/compatibility check at runtime;
- safe fallback to PyTorch SDPA;
- direct numeric and dispatched-kernel verification;
- no README claim from an untested environment.

## Precision

BF16/FP16 reduce tensor bytes and can activate Tensor Cores. FP16 may require scaling; BF16 has better
range. TF32 accelerates selected FP32 matrix multiplications with reduced mantissa precision.

Every precision experiment must record model, shape, backend, device, tolerances, loss trajectory,
throughput, and peak memory. A one-step finite output is not training-equivalence evidence.

## `torch.compile`

Compilation can fuse pointwise work, reduce Python overhead, and specialize graphs. It also has a
first-run compile cost, graph breaks, shape guards, cache growth concerns, and version/backend
sensitivity.

Planned comparison:

```text
eager warm-up -> repeated steady-state samples
compile first-call cost -> repeated steady-state samples
```

Compile will not be default merely because one steady-state case is faster. The project must state
the number of calls needed to amortize compilation and cases with regressions.

## Gradient checkpointing

Checkpointing discards selected forward activations and recomputes them in backward. It reduces
activation memory but increases compute and can interact with RNG/dropout. Tests must verify loss and
gradient tolerance and correct RNG preservation.

## Fused optimizer and accumulation

The baseline requests fused AdamW only on CUDA and records whether it was effectively enabled.
Gradient accumulation reduces peak activation memory but raises the number of microbatch launches.
Performance comparisons use the same effective number of target tokens per optimizer update.

## Distributed training

DDP replicates parameters and all-reduces gradients. FSDP shards parameters, gradients, and optimizer
state depending on strategy. Communication, overlap, topology, and checkpoint semantics dominate
at scale.

Windows distributed support and the single available GPU cannot provide credible scaling evidence.
The future distributed entry will target Linux and first prove single-process semantic equivalence.

## Benchmark protocol

Every promoted timing uses:

1. fixed seed and random/loaded model status;
2. exact batch, prompt, decode, dtype, backend, and parameter count;
3. correctness gate before timing;
4. warm-up;
5. synchronization before and after CUDA samples;
6. multiple raw samples and p50/p90/min/max/mean/stdev;
7. peak allocated memory definition;
8. actual environment and operator evidence;
9. negative cases and conclusion boundary.

Temperature, power limits, and background load are current residual uncertainties. One laptop result
is a case study, not a universal ranking.

