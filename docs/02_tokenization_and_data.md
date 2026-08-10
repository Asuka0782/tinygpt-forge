# Tokenization and data boundaries

## Tokens are the model's coordinate system

A language model does not consume text directly. A tokenizer maps text to integer IDs:

```text
text -> token IDs [N] -> training windows x,y [B,T]
```

The vocabulary determines output-head size, sequence length, OOV behavior, and what one “token/s”
means. Benchmarking token throughput across different tokenizers without also reporting text
compression can be misleading.

## Character baseline

`CharacterTokenizer` sorts the unique training code points and reserves `<unk>`. Its strengths are:

- deterministic construction;
- exact round-trip for in-vocabulary text;
- easy inspection and no compiled optional dependency;
- a transparent OOV count.

Its weaknesses are long sequences, poor reuse of multi-character units, and a large output space
for diverse Unicode corpora. It is a teaching and smoke-test baseline, not the final tokenizer for a
useful language model.

## Why split before fitting

Suppose the vocabulary is built from the full corpus before splitting. A character that appears only
in test data changes the vocabulary and therefore the model architecture. That leaks information from
test into preprocessing.

TinyGPT Forge first makes contiguous raw-text splits, then fits the tokenizer on train text only:

```text
raw text
  -> train text | validation text | test text
  -> fit tokenizer(train text)
  -> encode each split with the frozen tokenizer
```

Validation may select a checkpoint or hyperparameter. Test data is evaluated after choices are
fixed. Test loss is not a training dashboard metric.

## Shifted targets

Given token IDs `[x0,x1,x2,x3,x4]` and block size four:

```text
input  x = [x0,x1,x2,x3]
target y = [x1,x2,x3,x4]
```

Both tensors have shape `[B,T]`; logits have `[B,T,V]`. Cross-entropy flattens the first two
dimensions and compares every input position with exactly one next token.

The package intentionally exposes two APIs:

- `aligned_next_token_cross_entropy(logits, y)` for an already shifted `(x,y)` batch;
- `shifted_next_token_cross_entropy(logits, ids)` for one complete unshifted sequence.

Naming the convention prevents silent double shifting.

## Deterministic random windows

`NextTokenBatcher` owns a CPU `torch.Generator`. Sampling start positions on CPU makes its RNG state
portable across the training device. The state is stored in the resume checkpoint, so the next batch
after recovery is the same as the uninterrupted run.

This is different from the global Torch RNG used by dropout. Both must be restored for bitwise
continuation.

## BPE, Unigram, and future scope

Byte-pair encoding (BPE) repeatedly merges common adjacent units. Unigram tokenization starts with
a candidate vocabulary and removes units according to a probabilistic objective. Production choices
also include byte fallback, Unicode normalization, pre-tokenization, and special-token policy.

Before adding BPE/Unigram, this project requires:

- tokenizer trained only on the train split;
- versioned JSON/model artifacts and source license;
- encode/decode round-trip and unknown-byte behavior;
- vocabulary size, average tokens per code point/byte, OOV, and throughput comparison;
- deterministic reproduction from a frozen corpus hash and config.

## Data-system implications

For a tensor of integer token IDs, random window sampling is cheap. At scale, repeatedly decoding
text and constructing Python lists becomes the bottleneck. A future pipeline may use memory mapping
or streaming shards, but must record shard version, dtype, endianness, document boundaries, and
whether windows cross documents.

## Current limitations

- The bundled smoke corpus is tiny and repeated.
- Contiguous code-point splits can cut sentences.
- There is no document-level independence in the fixture.
- Character token/s cannot be compared directly with subword token/s as a measure of text speed.

