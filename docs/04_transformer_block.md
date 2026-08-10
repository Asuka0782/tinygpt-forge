# The modern TinyGPT decoder block

## Pre-normalized residual structure

TinyGPT Forge uses:

\[
h = x + \operatorname{Attention}(\operatorname{RMSNorm}(x)),
\]

\[
y = h + \operatorname{SwiGLU}(\operatorname{RMSNorm}(h)).
\]

Pre-normalization gives each sublayer a normalized input while the residual stream keeps an identity
path. It is usually easier to optimize deeply than the original post-normalized layout, but the
claim still depends on architecture and training conditions.

## RMSNorm

For a vector `x` of width `C`:

\[
\operatorname{RMSNorm}(x)=g\odot
\frac{x}{\sqrt{\frac{1}{C}\sum_i x_i^2+\epsilon}}.
\]

Unlike LayerNorm, RMSNorm does not subtract the mean or learn a bias in this implementation. It
controls magnitude with fewer reductions. FP16/BF16 inputs accumulate the square mean in FP32 to
avoid unnecessary underflow/rounding error, then cast back.

## RoPE

For adjacent channel pair `(x_2i,x_2i+1)` and angle `theta_{p,i}` at position `p`:

\[
\begin{bmatrix}x'_{2i}\\x'_{2i+1}\end{bmatrix}=
\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}
\begin{bmatrix}x_{2i}\\x_{2i+1}\end{bmatrix}.
\]

Rotation preserves pairwise norm. Applying it to Q and K makes their dot product depend on relative
position. Head dimension must be even. Large absolute positions in low precision can accumulate
phase error, so “supports arbitrary context” is not implied by the formula alone.

## MHA, MQA, and GQA

- MHA: `Hq = Hkv`; every query head has independent K/V.
- MQA: `Hkv = 1`; all query heads share one K/V head.
- GQA: `1 < Hkv < Hq`; groups of query heads share K/V.

Projection parameters and KV-cache bytes shrink with `Hkv`, but quality and kernel behavior may
change. The project validates shape and semantic execution; it does not claim one head configuration
is universally better.

For cache capacity `S`, element size `e`, and `L` layers, K+V storage is:

\[
2LBH_{kv}SD_he\quad\text{bytes}.
\]

## SwiGLU

The feed-forward path computes:

\[
\operatorname{SwiGLU}(x)=W_d(\operatorname{SiLU}(W_gx)\odot W_ux).
\]

It uses separate gate and value projections followed by an elementwise product. Compared with a
two-matrix GELU MLP, it changes both activation and parameter allocation; fair comparisons should
match total parameters or compute, not just `d_ff`.

## Tied language-model head

The output projection shares storage with the input token embedding when `tie_embeddings=true`.
This reduces parameters and couples input/output token representations. Parameter counting iterates
unique `nn.Parameter` objects so the shared matrix is counted once. Safetensors uses model-aware
save/load functions to preserve shared storage semantics.

## Initialization and dropout

Linear/embedding weights use a normal distribution with standard deviation 0.02. This is a simple
baseline, not a claim of optimal depth-scaled initialization. Dropout is configurable and disabled
in equivalence/benchmark configs; exact resumed training restores its global Torch RNG state.

## Engineering trade-offs

- Small, separate modules improve inspection and testing but add Python call boundaries.
- Explicit K/V repetition is portable but can allocate extra tensors; native GQA is a future
  backend-specific optimization.
- Float32 RoPE angle calculation for lower-precision activations improves stability but adds casts.
- Tied weights simplify the small model but complicate generic serialization unless shared storage
  is handled deliberately.
