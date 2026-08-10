# Line-by-line: components, RoPE, loss, cache, and generation

Notation: `B` is batch size, `T/Q/K/S` are sequence lengths, `C` is model width, `Hq/Hkv` are
query/KV head counts, `Dh=C/Hq`, `F` is feed-forward width, and `V` is vocabulary size.

## `model/components.py`

| Source lines | Explanation |
|---|---|
| [L1–L11](../../src/tinygpt_forge/model/components.py#L1) | The module docstring states the boundary. Postponed annotations keep Python 3.10-compatible type syntax from being eagerly evaluated. `cast`, PyTorch, `Tensor/nn`, and functional `F` are imports only: they allocate no tensors. `cast` exists solely because current PyTorch type stubs type `Module.__call__` broadly; it does not change runtime data. Blank lines separate standard-library, third-party, and declarations for Ruff. |
| [L12–L14](../../src/tinygpt_forge/model/components.py#L12) | `RMSNorm(nn.Module)` registers parameters and participates in `.to()`, `state_dict`, hooks, and autograd. RMS normalization omits mean subtraction: for one token vector (x\in\mathbb{R}^C), it uses (x/\sqrt{C^{-1}\sum_i x_i^2+\epsilon}). |
| [L15–L22](../../src/tinygpt_forge/model/components.py#L15) | The constructor first initializes `nn.Module`, rejects nonsensical dimensions/epsilon, stores the scalar, and creates a learned scale `[C]` initialized to one. Broadcasting later applies it over `[B,T,C]`. A plain tensor would not be optimized or serialized; `nn.Parameter` is required. |
| [L23–L25](../../src/tinygpt_forge/model/components.py#L23) | `forward` is the module call contract. Any leading dimensions are allowed as long as the final dimension matches the learned weight; the normal model supplies `[B,T,C]`. The docstring records the mixed-precision rule. |
| [L26–L30](../../src/tinygpt_forge/model/components.py#L26) | Half/BF16 inputs are promoted to FP32 for squaring, mean, and reciprocal square root, reducing overflow/rounding risk. `keepdim=True` produces `[B,T,1]`, so the scale broadcasts across `C`. The normalized tensor returns to the activation dtype before multiplying `[C]`. Autograd records these casts and reductions. This reads/writes (O(BTC)) elements; for tiny tensors promotion overhead can outweigh stability benefits, but for training it is the safer default. |
| [L31–L35](../../src/tinygpt_forge/model/components.py#L31) | Blank separation precedes `SwiGLU`. The gated feed-forward replaces a single activation branch with `SiLU(gate(x)) * up(x)`, allowing content-dependent multiplicative gating. |
| [L36–L40](../../src/tinygpt_forge/model/components.py#L36) | Three bias-free linears map `[B,T,C]→[B,T,F]`, `[B,T,C]→[B,T,F]`, then `[B,T,F]→[B,T,C]`. Biases are omitted to match the chosen modern decoder contract, not because bias is universally wrong. Parameter/FLOP cost is roughly `3*C*F` per token. |
| [L41–L46](../../src/tinygpt_forge/model/components.py#L41) | The forward expression keeps the two `[B,T,F]` branches shape-aligned, applies SiLU to the gate only, multiplies elementwise, and projects back to the residual width. Calling modules with `(...)` preserves hooks and compilation semantics; `cast(Tensor, …)` repairs only the static type. A naïve Python loop over tokens is mathematically equivalent but destroys GEMM batching. |

## `model/rope.py`

| Source lines | Explanation |
|---|---|
| [L1–L8](../../src/tinygpt_forge/model/rope.py#L1) | Module/import lines establish a PyTorch module with no learned parameters. RoPE will transform Q/K activations rather than add a position embedding to the residual stream. |
| [L9–L13](../../src/tinygpt_forge/model/rope.py#L9) | `RotaryEmbedding` consumes `[B,H,T,Dh]`. The class annotation tells Mypy that `register_buffer` creates `inv_freq`; a buffer follows device moves but is not optimized. |
| [L14–L22](../../src/tinygpt_forge/model/rope.py#L14) | `Dh` must be positive/even because coordinates are paired. Frequencies are (\theta_j=b^{-2j/D_h}) for `j=0…Dh/2-1`, stored in FP64 for stable recomputation. `persistent=False` omits this deterministic tensor from checkpoints, reducing artifact coupling; it is recreated from config on load. |
| [L23–L25](../../src/tinygpt_forge/model/rope.py#L23) | `forward` accepts explicit absolute positions. Cached decoding depends on this: the next token must use position `past_length`, not restart at zero. |
| [L26–L33](../../src/tinygpt_forge/model/rope.py#L26) | Shape validation rejects incorrect rank/`Dh`. When absent, positions are `0…T-1` on the activation device. An explicit position vector must have exactly `T` elements, preventing accidental batch/head broadcasting. |
| [L34–L41](../../src/tinygpt_forge/model/rope.py#L34) | Angle math uses FP32 except true FP64 inputs. `[T,1] * [1,Dh/2]` yields angles `[T,Dh/2]`; sine/cosine become `[1,1,T,Dh/2]` for batch/head broadcasting and are cast to activation dtype. Trig computation is (O(TD_h)), normally much smaller than attention's (O(T^2D_h)). |
| [L42–L47](../../src/tinygpt_forge/model/rope.py#L42) | Even/odd slices each have `[B,H,T,Dh/2]`. The pair rotation computes `(x_even*cos - x_odd*sin, x_even*sin + x_odd*cos)`, stacks a final size-2 axis, then flattens back to `[B,H,T,Dh]`. It preserves each pair's norm up to floating error; property tests verify that invariant. |

## `losses.py`

| Source lines | Explanation |
|---|---|
| [L1–L8](../../src/tinygpt_forge/losses.py#L1) | The module isolates the project's most error-prone convention: whether targets are already shifted. Imports have no model state. |
| [L9–L14](../../src/tinygpt_forge/losses.py#L9) | The aligned function requires logits `[B,T,V]` and targets `[B,T]`; `*` makes `ignore_index` keyword-only so a positional integer cannot be mistaken for another tensor. |
| [L15–L21](../../src/tinygpt_forge/losses.py#L15) | The docstring is part of the semantic API: each target is already “the token after this input position.” Training batches are constructed in this convention. |
| [L22–L29](../../src/tinygpt_forge/losses.py#L22) | Rank and prefix-shape checks fail before flattening. Without them, `reshape` could silently pair unrelated tokens or CrossEntropy could report a harder-to-diagnose size error. Target integer dtype/range is delegated to PyTorch, which produces the authoritative error. |
| [L30–L34](../../src/tinygpt_forge/losses.py#L30) | Flattening maps `[B,T,V]→[B*T,V]` and `[B,T]→[B*T]`; cross-entropy then performs log-softmax plus negative log likelihood efficiently. Complexity and saved activations are (O(BTV)). `reshape` handles non-contiguous input by copying only when necessary. |
| [L35–L36](../../src/tinygpt_forge/losses.py#L35) | Blank lines separate the second public convention. |
| [L37–L43](../../src/tinygpt_forge/losses.py#L37) | The shifted function accepts one unshifted sequence: token `t` is input and token `t+1` is label. Keeping it separate makes double-shifting visible in code review. |
| [L44–L50](../../src/tinygpt_forge/losses.py#L44) | It repeats rank/shape validation and requires at least two positions, because no next-token pair exists for `T=1`. |
| [L51–L55](../../src/tinygpt_forge/losses.py#L51) | Final logits are dropped and the first token label is dropped: `[B,T-1,V]` is aligned with `[B,T-1]`. Delegating to one implementation prevents the two APIs from drifting. |

## `cache.py`

| Source lines | Explanation |
|---|---|
| [L1–L10](../../src/tinygpt_forge/cache.py#L1) | The module is deliberately independent of attention math: it owns storage and validates writes. Imports identify Tensor and immutable model geometry. |
| [L11–L18](../../src/tinygpt_forge/cache.py#L11) | The class stores raw GQA K/V as `[B,Hkv,capacity,Dh]`, not repeated `[B,Hq,…]`. One shared logical length is advanced only after all layers write, preventing layer positions from diverging. The object is mutable per generation request and is not thread-safe. |
| [L19–L27](../../src/tinygpt_forge/cache.py#L19) | Keyword-only allocation arguments make batch/capacity/device/dtype explicit. Capacity is a semantic upper bound, not automatic growth; preallocation trades fixed memory for stable addresses and no per-step `cat`. |
| [L28–L33](../../src/tinygpt_forge/cache.py#L28) | Validation rejects empty batches, capacity beyond the model context, and integer cache storage. These errors occur before any large allocation. |
| [L34–L47](../../src/tinygpt_forge/cache.py#L34) | Geometry is recorded, device text is normalized by the first actual allocation, and one uninitialized K plus V tensor is allocated per layer. `torch.empty` avoids zero-fill because only the logical prefix is exposed. Storage is (2·L·B·H_{kv}·capacity·D_h·bytes(dtype)). Old bytes beyond `length` are inaccessible but not securely erased. |
| [L48–L57](../../src/tinygpt_forge/cache.py#L48) | `update` receives one layer's new `[B,Hkv,Tnew,Dh]` states and an expected absolute start. It returns views covering `0:end`, which attention reads immediately. |
| [L58–L64](../../src/tinygpt_forge/cache.py#L58) | Layer and start checks catch cross-layer misuse and stale callers. Requiring `start_position == length` makes every model call append-only unless the owner explicitly rewinds. |
| [L65–L78](../../src/tinygpt_forge/cache.py#L65) | One compound guard verifies rank, K/V equality, batch/KV heads, `Dh`, device, and dtype for both tensors. Silent dtype conversion during `copy_` would hide mixed-precision bugs, so it is rejected. The error intentionally names the whole contract rather than leaking tensor contents. |
| [L79–L87](../../src/tinygpt_forge/cache.py#L79) | `end=start+Tnew` is checked against capacity. In-place `copy_` writes only the new slice; returned prefix views share storage and allocate no K/V copy. A Dynamic Cache instead performs `torch.cat`, allocating and copying an ever-growing prefix. |
| [L88–L96](../../src/tinygpt_forge/cache.py#L88) | `advance` is the commit step after all layers succeed. Positive/capacity checks protect logical state; only an integer addition changes visibility. If a layer fails before this call, the next attempt overwrites the same uncommitted range. |
| [L97–L102](../../src/tinygpt_forge/cache.py#L97) | `reset` sets length to zero in (O(1)); it does not zero memory or free VRAM. This is performance behavior, not a secure data-erasure API. |
| [L103–L109](../../src/tinygpt_forge/cache.py#L103) | `rewind` can retain any existing prefix `0…length`, used by steady-state benchmarks. Moving forward is forbidden because unwritten bytes would become visible. |
| [L110–L117](../../src/tinygpt_forge/cache.py#L110) | The property sums actual tensor element counts times element sizes over all K/V layers. It reports allocated tensor bytes, not allocator reservation, Python overhead, or temporary attention workspace. |
| [L118–L123](../../src/tinygpt_forge/cache.py#L118) | `layer_storage` exposes full-capacity tensors only for tests/memory inspection after validating the layer. Production attention uses prefix views from `update`; exposing this method documents, rather than hides, the storage contract. |

## `generation.py`

| Source lines | Explanation |
|---|---|
| [L1–L14](../../src/tinygpt_forge/generation.py#L1) | Imports assemble sampling, cache, cache tuple type, and model without networking or tokenizer assumptions. Generation works on integer IDs, keeping text policy outside the kernel path. |
| [L15–L23](../../src/tinygpt_forge/generation.py#L15) | A frozen dataclass makes one immutable sampling contract. `temperature=0` means greedy; `top_k=None` means no truncation; seed belongs to generation, not global training RNG. |
| [L24–L30](../../src/tinygpt_forge/generation.py#L24) | Validation allows zero new tokens, rejects negative temperature, and requires positive `top_k`. `top_k>V` is legal and later clamps to `V`. |
| [L31–L36](../../src/tinygpt_forge/generation.py#L31) | `sample_next_token` consumes the final-position logits `[B,V]` and a device-compatible generator. Returning `[B,1]` is deliberate so it concatenates along sequence dimension. |
| [L37–L42](../../src/tinygpt_forge/generation.py#L37) | Shape validation precedes sampling. At zero temperature, `argmax` is deterministic and ignores RNG; no division by zero or softmax is performed. |
| [L43–L49](../../src/tinygpt_forge/generation.py#L43) | Positive temperature rescales logits. Top-k obtains the kth threshold per batch and masks smaller logits to `-inf`; ties at the threshold can retain more than exactly `k`, a common stable convention. Softmax yields `[B,V]`, and multinomial returns `[B,1]`. Sampling complexity is (O(BV)). |
| [L50–L62](../../src/tinygpt_forge/generation.py#L50) | `@torch.inference_mode()` disables gradient recording and version-counter overhead for the whole call. Keyword-only backend/cache controls prevent positional confusion. The same function owns full, Dynamic, and Static paths so sampling semantics cannot diverge. |
| [L63–L69](../../src/tinygpt_forge/generation.py#L63) | Prompt rank/non-empty, total context capacity, and cache implementation are checked before allocating cache or launching the model. `max_new_tokens=0` is valid and returns the prompt unchanged. |
| [L70–L85](../../src/tinygpt_forge/generation.py#L70) | `generated` initially aliases input IDs; concatenation later creates new tensors without modifying the caller. A generator on the input device is seeded locally. Dynamic cache starts as `None`. Static mode derives activation dtype from model parameters and allocates exactly prompt plus decode capacity. `step_input` is the whole prompt only for prefill. |
| [L86–L101](../../src/tinygpt_forge/generation.py#L86) | Each decode step either feeds `step_input` with cache or recomputes over all `generated`. On the first cached call `step_input` is `[B,prompt]`; later it is `[B,1]`. Dynamic mode requires returned K/V tuples; Static mode mutates owned storage and returns no tuple. Full mode provides the correctness baseline but repeats prefix projections/attention. |
| [L102–L107](../../src/tinygpt_forge/generation.py#L102) | Only final logits `[B,V]` are sampled. New IDs are concatenated to `[B,current+1]`, and the new token becomes next cached input. After exactly `max_new_tokens` iterations, the function returns prompt plus continuation. Python-loop and concatenation overhead explain why algorithmically cheaper caching can lose on tiny GPU shapes. |

