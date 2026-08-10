# From n-grams to a decoder-only Transformer

## The recurring question

Language modeling repeatedly asks one question:

\[
p(x_1,\ldots,x_T)=\prod_{t=1}^{T}p(x_t\mid x_{<t}).
\]

The history is largely about how to represent `x_<t`, how far back to look, and how to train the
representation efficiently.

## Unigram and bigram

A unigram ignores context and estimates `p(x_t)`. A bigram keeps only the previous token:

\[
p(x_t\mid x_{<t})\approx p(x_t\mid x_{t-1}).
\]

For a vocabulary of size `V`, a direct neural bigram table has `V × V` logits. It is easy to inspect
and provides a valuable baseline, but it cannot distinguish two prefixes ending in the same token.

What it teaches:

- negative log-likelihood and perplexity have concrete meanings;
- adding parameters does not guarantee better validation loss;
- a baseline must use the same split and metric as later models.

## Fixed context averaging

A natural next idea embeds the previous `T` tokens and averages their vectors. It uses more context,
but every position receives the same content-independent weight. Word order is lost unless separate
position information is added.

This stage explains why “more context” is not enough. A useful model also needs a way to choose
which context is relevant to the current query.

## Single-head causal self-attention

Self-attention creates a content-dependent weighted average:

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
\]

\[
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right),\qquad Y=AV.
\]

The causal mask `M` sets future positions to negative infinity before softmax. The scale
`1/sqrt(d_h)` prevents dot-product variance from growing with the head dimension and pushing
softmax into saturated regions.

Unlike fixed averaging, attention can assign different weights for every query position and input
sequence. Its main cost is the `[T,T]` score matrix.

## Multi-head attention

With `H` heads, the model learns several lower-dimensional attention patterns in parallel:

\[
\operatorname{MHA}(X)=\operatorname{Concat}(Y_1,\ldots,Y_H)W_O.
\]

Holding `d_model` fixed can keep projection parameter count roughly constant while changing the
number and dimension of heads. More heads are not automatically better: the result depends on data,
optimization, head dimension, and randomness.

## Transformer block and depth

Attention mixes information across positions. A feed-forward network transforms each position
independently. Residual paths and normalization make many such blocks trainable:

\[
X' = X + \operatorname{Attention}(\operatorname{Norm}(X)),
\]

\[
X'' = X' + \operatorname{FFN}(\operatorname{Norm}(X')).
\]

Depth allows later layers to build on earlier contextual features. It also adds optimization and
systems costs, so fair depth comparisons need matched data, initialization policy, training budget,
and multiple seeds.

## RoPE, GQA, and KV Cache

Learned absolute position embeddings attach one vector to each configured index. Rotary position
embeddings (RoPE) rotate query/key pairs by position-dependent angles; their dot product naturally
depends on relative displacement.

During autoregressive generation, a full implementation repeatedly recomputes old keys and values.
A KV Cache stores them. Grouped-query attention (GQA) lets several query heads share one key/value
head, reducing cache storage from a factor of `Hq` to `Hkv`.

Algorithmically, caching removes repeated prefix projections and attention work. Systems behavior
still depends on batch size and GPU utilization. This repository's batch-one cases were slower with
cache, while a batch-eight case crossed over and became faster.

## Hugging Face, LoRA, and the ecosystem boundary

Modern frameworks add standardized configs, tokenizers, model hubs, distributed training, adapter
fine-tuning, safe serialization, and optimized serving. TinyGPT Forge does not reimplement all of
them. It builds a small semantic core that can be checked by hand, then adds only mechanisms that
have local correctness and performance evidence.

The progression is therefore not “old code replaced by fashionable code.” Each stage answers the
failure exposed by the previous one:

```text
no context
  -> one-token context
  -> fixed multi-token context
  -> content-dependent causal context
  -> multiple attention subspaces
  -> deep residual computation
  -> relative position structure
  -> reuse during autoregressive decoding
  -> ecosystem interoperability and parameter-efficient tuning
```

## Check your understanding

1. Why can a context-average model underperform a bigram despite reading more tokens?
2. What numerical problem does `1/sqrt(d_h)` address?
3. Why is a lower best validation loss from an unmatched checkpoint pair not causal evidence for
   RoPE?
4. How can KV Cache reduce FLOPs and still increase latency?

Answers and common misconceptions are in `12_exercises_and_answers.md`.

