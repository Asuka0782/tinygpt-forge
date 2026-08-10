# Exercises and answers

Try each question before opening the answer beneath it.

## 1. Shifted targets

Tokens are `[4,7,2,9,3]` and block size is four. Write `x` and `y`.

**Answer:** `x=[4,7,2,9]`, `y=[7,2,9,3]`. Every `x[t]` predicts `y[t]`; do not shift again inside
an aligned-loss function.

Common mistake: using `x=[4,7,2,9]`, `y=[2,9,3]`, which drops two transitions or causes shape hacks.

## 2. Why more context can hurt

Why may uniform context averaging underperform a bigram?

**Answer:** It destroys order and forces every context position to have equal weight. Irrelevant
tokens dilute the predictive previous token. More available information is not useful unless the
model can select/represent it.

## 3. Attention scaling

If independent Q/K components have variance one, what happens to dot-product variance as `Dh`
grows, and why divide by `sqrt(Dh)`?

**Answer:** The sum of `Dh` products has variance proportional to `Dh`. Division makes the score
scale roughly dimension-independent, reducing softmax saturation and tiny gradients.

## 4. Causal gradient test

A loss uses only logits at position three. Which input embedding positions must have zero gradient?

**Answer:** positions four and later. Positions zero through three may contribute. A zero upper
attention matrix alone does not prove there is no other future path; the end-to-end gradient test
does.

## 5. GQA cache bytes

For `L=8`, `B=4`, `Hkv=2`, `S=128`, `Dh=64`, BF16 (`e=2` bytes), compute K+V cache bytes.

**Answer:** `2×8×4×2×128×64×2 = 2,097,152` bytes = 2 MiB. Using `Hq=8` instead would incorrectly
inflate storage fourfold.

## 6. Cached mask

Past length is four and current query length is two. Which key indices may query rows zero and one
read?

**Answer:** current absolute positions are four and five. Row zero reads keys `0..4`; row one reads
`0..5`. A naïve two-row top-left triangular mask would be wrong.

## 7. RoPE invariance

Why do Q/K dot products remain unchanged if the same constant is added to every position?

**Answer:** Rotation composition gives `R(p+c)^T R(q+c)=R(q-p)` for compatible pairwise rotations;
the common shift cancels, leaving relative displacement. Floating-point error prevents perfect
identity at large/low-precision positions.

## 8. Resume state

Model weights and learning rate are restored, but Adam moments and batch RNG are not. Is this exact
resume?

**Answer:** No. The optimizer update direction/magnitude and next sampled windows change. It is a
new trajectory starting from old weights.

## 9. Gradient accumulation

Why divide each microbatch loss by `K` when accumulating `K` equal-sized microbatches?

**Answer:** Backward sums gradients. Division makes the accumulated gradient equal the mean over the
effective batch. Without it, gradient magnitude is `K` times larger.

## 10. KV Cache negative result

How can cache perform fewer FLOPs but run slower at batch one?

**Answer:** One-token kernels underutilize the GPU; Python dispatch, explicit masks, small GEMMs,
allocations/copies, and launch latency dominate. Full-sequence work has more arithmetic but much
better parallel efficiency.

## 11. Static versus Dynamic

Does Static Cache always use less memory and run faster?

**Answer:** No. It avoids growing concatenations and provides fixed storage, but preallocates full
capacity and performs writes. End-to-end peak memory can rise, and kernel utilization can dominate.

## 12. SDPA versus FlashAttention

All SDPA backend flags are enabled. Can README say FlashAttention was used?

**Answer:** No. Flags describe permission/availability choices, not necessarily compiled support or
actual dispatch. Inspect profiler operator/kernel evidence and forced-backend behavior.

## 13. Validation and test

Can test loss choose the best checkpoint?

**Answer:** No. That turns the test set into validation data and biases final reporting. Validation
selects; test evaluates fixed choices.

## 14. LoRA first gradient

With random `A` and zero `B`, why may `grad(A)=0` but `grad(B)≠0` on the first step?

**Answer:** The update is `BAx`. Differentiation with respect to A contains B, which is zero;
differentiation with respect to B contains `Ax`, which is generally nonzero.

## 15. Design task

Propose a fair MHA/GQA comparison.

**Answer:** Freeze data/splits/tokenizer, seed policy, total parameter or compute budget, training
tokens, optimizer, dtype/backend, evaluation windows, and generation settings. Report multiple seeds
or narrow the claim. Measure loss/perplexity plus prefill/decode/cache bytes; do not compare only
head count with unconstrained parameter changes.

## 16. Coding task

Modify a copy of the manual attention mask to allow one future token. Which tests should fail?

**Answer:** manual/SDPA equivalence, upper-triangle probability zero, suffix-invariance, and future
gradient tests should fail. If only one fails, inspect whether the others cover the altered row/shape.

## 17. Experiment task

What is the highest-information next compile experiment?

**Answer:** Use a fixed model and shape grid, record eager correctness/timing, first compile latency,
graph breaks, compiled steady-state raw samples, and the call count required to amortize compilation.
Include a shape where compile loses.

