# Benchmarks: method before headline

## What these measurements answer

The current benchmark asks two narrow questions:

1. For one forward prefill, how does the readable attention path compare with PyTorch SDPA?
2. For greedy generation, when do concat-based Dynamic and preallocated Static KV Caches beat
   recomputing the full sequence?

It does **not** measure language quality, production serving throughput, or performance on other
GPUs. The model weights are random so the comparison isolates execution mechanisms.

## Measurement protocol

- Device: NVIDIA GeForce RTX 5060 Laptop GPU, compute capability 12.0, approximately 8 GiB.
- Software: Windows 11, Python 3.14.3, PyTorch 2.11.0+cu128.
- Dtype: FP32; tested batch sizes are 1 and 8, as recorded per artifact.
- Every CUDA sample is synchronized before and after timing.
- Each case has warm-up calls and stores every raw repeated sample in JSON.
- Decode is end-to-end greedy generation, including Python dispatch and token concatenation.
- Correctness gates run before timing: manual/SDPA logits tolerance and full/cache greedy token
  identity.
- The harness rejects non-finite errors or max-absolute error above `5e-5` for FP32 and `5e-2` for
  BF16/FP16; each new artifact records the selected threshold.
- The profiler reports `aten::_scaled_dot_product_efficient_attention`; this PyTorch wheel reports
  FlashAttention as not compiled.

The raw artifacts are in `results/benchmarks/`. Profiler initialization warns that CUDA CUPTI
activities are unavailable on this Windows setup; CPU-side operator names still reveal the
dispatched ATen operation, but kernel-level CUDA timelines are not claimed.

![Recorded RTX 5060 p50 speedups](../figures/fig_rtx5060_speedups.png)

The plot is generated directly from the three `*static_v3.json` artifacts by
[`figures/gen_fig_rtx5060_speedups.py`](../figures/gen_fig_rtx5060_speedups.py); a
[vector PDF](../figures/fig_rtx5060_speedups.pdf) is checked in for print-quality rendering. Values
below the dashed `1.0×` line are slower than the relevant baseline and are intentionally retained.
To regenerate it from a clean checkout, install the non-runtime plotting extra and run:

```bash
python -m pip install -e ".[plots]"
python figures/gen_fig_rtx5060_speedups.py
```

## Stage-3 Dynamic Cache baseline

| Parameters | Prompt + decode | Manual prefill p50 | SDPA prefill p50 | SDPA speedup | Full decode p50 | Dynamic cache p50 | Cache speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 106,816 | 16 + 16 | 1.659 ms | 1.357 ms | 1.223× | 24.239 ms | 34.982 ms | 0.693× |
| 106,816 | 32 + 32 | 1.786 ms | 1.486 ms | 1.201× | 63.639 ms | 68.697 ms | 0.926× |
| 3,148,032 | 128 + 64 | 3.302 ms | 2.704 ms | 1.221× | 239.390 ms | 262.345 ms | 0.913× |

All three cases produced identical full/cache greedy IDs. Manual/SDPA maximum absolute logit
error ranged from `2.38e-7` to `8.05e-7`.

For 16 + 16, full and dynamic-cache decode achieved approximately 660 and 457 generated tokens/s.
For 128 + 64, they achieved approximately 267 and 244 tokens/s. The medium case reduced measured
peak allocated memory from about 23.44 MiB to 22.52 MiB, but that small difference is specific to
this random model and allocator state.

## Interpretation

SDPA is consistently around 1.20—1.22× faster than the explicit attention matrix for these three
prefill shapes. That is evidence for these shapes on this environment, not a universal ratio.

The current Dynamic KV Cache is slower in every measured decode case. This does **not** refute the
algorithmic benefit of KV caching. The full path repeats prefix projections and attention, whereas
the cached path avoids that work; however, this implementation concatenates K/V tensors at every
step, launches many very small `q_len=1` kernels, and pays Python overhead. Efficient full-sequence
kernels parallelize the tiny workload well enough to win at the tested sizes.

The result therefore rejects the claim “adding any KV Cache automatically makes a tiny model
faster.” It motivated a preallocated Static Cache and a separate steady-state single-token metric.

## Static Cache and crossover experiment

Static Cache writes in place to `[B, Hkv, capacity, Dh]`. The benchmark now compares full,
concat-based Dynamic, and preallocated Static paths, and measures both complete generation and
one model call after a prefilled prompt.

| Parameters | Batch | Prompt + decode | Dynamic end-to-end | Static end-to-end | Dynamic single-token | Static single-token |
|---:|---:|---:|---:|---:|---:|---:|
| 3,148,032 | 1 | 128 + 64 | 0.915× | 0.931× | 0.921× | 0.933× |
| 24,650,240 | 1 | 128 + 16 | 0.923× | 0.887× | 0.920× | 0.908× |
| 24,650,240 | 8 | 128 + 8 | **1.331×** | **1.319×** | **1.348×** | **1.389×** |

Every row passed full/Dynamic/Static greedy-token identity. Single-token logit errors were below
`2e-6`. Each row used warm-up and repeated synchronized samples; see the corresponding `*v3.json`
files for raw values.

The crossover appeared when batch size increased to eight. At batch one, full-sequence GEMMs and
SDPA use the GPU efficiently enough that many tiny `q_len=1` calls lose despite doing fewer
operations. At batch eight, cached decode has enough parallel work and becomes faster. This is a
hardware/shape boundary, not a universal batch-eight rule.

Static Cache is not consistently faster than Dynamic Cache here. It avoids repeated K/V
concatenation, but end-to-end measurement includes allocation and in-place writes; kernel launch
and attention backend behavior remain dominant. In the batch-eight row, Static has the best
single-token speedup but is slightly slower end-to-end than Dynamic.

Peak allocated memory also needs careful wording. Cache storage reduces repeated intermediate
work, but the end-to-end benchmark holds explicit cache tensors. In the batch-eight row, measured
peak allocation was about 166.0 MiB for full decode and 168.4/168.9 MiB for Dynamic/Static. The
medium batch-one row instead reduced peak allocation by roughly 0.8—0.9 MiB. Neither is a general
memory ratio.

## Next highest-information experiment

Separate time-to-first-token from steady-state decode across a systematic `(batch, prompt,
decode, dtype)` grid, then test whether `torch.compile` or a backend with efficient native GQA
changes the crossover. Promotion still requires:

- identical greedy IDs and bounded per-step logit error;
- actual dispatched operator/kernel evidence;
- raw synchronized samples rather than one aggregate number;
- explicit model, dtype, device, power/thermal limitations.
