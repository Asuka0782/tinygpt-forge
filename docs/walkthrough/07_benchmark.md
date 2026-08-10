# Line-by-line: correctness-gated benchmarks

The benchmark is an experiment harness, not a generic performance verdict. It first checks that the
implementations agree, then measures each path with the same model, inputs, shape, device, dtype,
warm-up count, repeat count, and synchronization rule. Raw samples remain in the JSON so summaries
can be independently recomputed.

## `benchmark.py`

| Source lines | Explanation |
|---|---|
| [L1–L24](../../src/tinygpt_forge/benchmark.py#L1) | Imports cover JSON/evidence, statistics/timing, environment metadata, PyTorch profiling, and the exact cache/model/generation implementations under test. There is no training or network dependency. |
| [L25–L29](../../src/tinygpt_forge/benchmark.py#L25) | CUDA launches are asynchronous, so `_synchronize` blocks only on CUDA. Without a pre/post barrier, a CPU timer could measure dispatch rather than completed GPU work. CPU operations are already synchronous for this use. |
| [L30–L35](../../src/tinygpt_forge/benchmark.py#L30) | The percentile uses the nearest-rank rule on sorted samples: index `ceil(qN)-1`, clamped at zero. For small `N`, p90 is coarse; therefore raw samples and repeat count are more informative than one percentile. |
| [L36–L49](../../src/tinygpt_forge/benchmark.py#L36) | `_summarize` retains unit, count, arithmetic mean, sample standard deviation, median p50, nearest-rank p90, extrema, peak allocated memory, and every sample. It does not hide outliers or report confidence intervals. |
| [L50–L64](../../src/tinygpt_forge/benchmark.py#L50) | `_measure` accepts a zero-argument tensor computation. Warm-up triggers lazy kernel/library setup outside timing, then synchronization drains work and peak-memory tracking resets after warm-up. Peak means PyTorch allocated bytes, not total board usage. |
| [L65–L76](../../src/tinygpt_forge/benchmark.py#L65) | Every repeat synchronizes, starts a nanosecond host clock, executes, synchronizes again, and stores milliseconds. Touching `numel` rejects an invalid empty result and keeps a Python reference through completion. The returned summary includes all runs, not only the best. |
| [L77–L91](../../src/tinygpt_forge/benchmark.py#L77) | A short PyTorch profiler pass records operator keys containing SDPA/flash/efficient-attention markers. CPU profiler activity observes dispatched ATen operators even for CUDA work; this is backend evidence, not a proof that an external FlashAttention package is installed. |
| [L92–L106](../../src/tinygpt_forge/benchmark.py#L92) | `run_benchmark` makes model contract, device, dtype, `[B,T]` shape, decode length, warm-up/repeats, and seed explicit. The return value is a serializable evidence document. |
| [L107–L120](../../src/tinygpt_forge/benchmark.py#L107) | Validation rejects unavailable CUDA, unsupported dtype/CPU FP16, nonpositive shape/decode/repeat values, negative warm-up, and sequences beyond model context. A benchmark that cannot represent the requested computation fails before allocation. |
| [L121–L138](../../src/tinygpt_forge/benchmark.py#L121) | Dtype is mapped to a PyTorch scalar type; CPU/all-CUDA RNGs are seeded before random weights and `[B,T]` token IDs. The eval model is converted to the measured dtype. CUDA properties are captured once. Greedy generation (`temperature=0`) removes sampling noise. |
| [L139–L155](../../src/tinygpt_forge/benchmark.py#L139) | Under inference mode, identical input and weights produce manual and SDPA logits `[B,T,V]`. Dynamic prefill yields layer KV tensors, the SDPA last logits select `[B,1]` next IDs, and full recomputation versus one cached-token step produce comparable `[B,V]` logits. |
| [L156–L170](../../src/tinygpt_forge/benchmark.py#L156) | A static cache with capacity `T+1` and model parameter dtype/device is prefilled, then receives the same next token. This isolates static write/index semantics from generation-loop effects before timing. |
| [L171–L193](../../src/tinygpt_forge/benchmark.py#L171) | Full, dynamic-cache, and static-cache greedy generation each produce `[B,T+D]` IDs from the same prompt/config. These runs are correctness gates and warm the broad paths; their outputs are not timed samples. |
| [L194–L212](../../src/tinygpt_forge/benchmark.py#L194) | After synchronization, max absolute FP32-comparison errors are computed for manual/SDPA and both cached one-token paths. Tolerance is `5e-5` for FP32 and `5e-2` for BF16/FP16. Non-finite/excess error or any greedy-ID divergence aborts before a speed number can be reported. |
| [L213–L223](../../src/tinygpt_forge/benchmark.py#L213) | Static position rewinds to the post-prefill boundary. Each steady-state timed call rewinds again, overwrites the same next slot, and returns logits. Allocation and prefill are excluded from this specific metric. |
| [L224–L249](../../src/tinygpt_forge/benchmark.py#L224) | With gradients disabled, manual and SDPA prefill measure full `[B,T]→[B,T,V]` forward calls. Full decode measures the complete Python greedy loop with repeated full-prefix attention and token concatenation. |
| [L250–L275](../../src/tinygpt_forge/benchmark.py#L250) | Dynamic and static end-to-end decode use the same loop and prompt, changing only cache implementation. Dynamic concatenation and static allocation are included, so a result below `1×` is retained rather than hidden. |
| [L276–L300](../../src/tinygpt_forge/benchmark.py#L276) | Three one-call metrics compare full-prefix recomputation, an already-prefilled dynamic cache, and an already-allocated/prefilled static cache. This estimates steady-state decode kernel overhead separately from end-to-end allocation, Python sampling, and concatenation. |
| [L301–L320](../../src/tinygpt_forge/benchmark.py#L301) | Median denominators are named once. FlashAttention availability is capability-probed because PyTorch 2.3 lacks the later helper; an absent helper safely records `false`. The v2 evidence schema records UTC timestamp, seed, random initialization, parameter count, exact model config, and shape. Filename revisions are experiment iterations; `format` is the JSON schema version. |
| [L321–L329](../../src/tinygpt_forge/benchmark.py#L321) | Method metadata states warm-up/repeats, synchronization, and the two decode scopes. Comparing end-to-end cache numbers with single-token numbers without this distinction would be misleading. |
| [L330–L360](../../src/tinygpt_forge/benchmark.py#L330) | Environment records Python, PyTorch, OS, device/name, dtype, CUDA runtime, cuDNN, compute capability, board memory, SM count, compiled flash availability, and observed SDPA operator names. PyTorch exposes no stable public driver-version call here; the machine audit records the NVIDIA driver separately. A narrow type-ignore documents PyTorch's untyped public cuDNN accessor. CPU uses explicit null/not-profiled values. |
| [L361–L368](../../src/tinygpt_forge/benchmark.py#L361) | The exact correctness values and tolerance accompany timings. A consumer can verify both token identity and floating-point error instead of trusting an unrecorded precondition. |
| [L369–L396](../../src/tinygpt_forge/benchmark.py#L369) | All eight timing summaries are embedded. Speedup is `baseline_p50 / candidate_p50`, so `>1` means faster and `<1` means regression. End-to-end throughput is `B·D / (p50_ms/1000)` generated tokens/s; it must be compared only at the same shape and scope. |
| [L397–L405](../../src/tinygpt_forge/benchmark.py#L397) | Limitations explicitly bound interpretation: random weights test mechanisms, one machine/shape is not universal, dynamic concatenation can lose on tiny work, static allocation is included end-to-end, and system power/temperature/background load are uncontrolled. The complete document then returns. |
| [L406–L418](../../src/tinygpt_forge/benchmark.py#L406) | `save_benchmark` creates parent directories, writes formatted UTF-8 JSON with a final newline to a sibling temporary file, then atomically replaces the destination. A crash cannot leave a half-written file at the advertised result path. |

## What the numbers do and do not mean

The harness measures implementation paths for a fixed synthetic shape. It does not measure model
quality, multi-request serving, power efficiency, thermal stability, or confidence across machines.
A feature is accepted only when correctness passes; whether it is faster remains an empirical,
shape-dependent result.
