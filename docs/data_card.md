# Data card: bundled smoke corpus

## Source and license status

`examples/tiny_corpus.txt` was written specifically for this repository. It describes the project's
own mechanisms and does not copy the original private learning corpus. It is distributed under the
repository's Apache-2.0 license.

No Day16—Day23 corpus, Hugging Face cache, external dataset, private prompt, or model checkpoint is
included in the repository.

## Intended use

The file is a deterministic fixture for exercising:

- UTF-8 loading and character vocabulary fitting;
- train/validation/test responsibility boundaries;
- next-token batching and shifted targets;
- short CPU training, checkpointing, resume, and generation.

It is not large or diverse enough for language-quality evaluation, fairness analysis, or downstream
task claims. Repetition in generated text is expected.

## Processing

Raw text is split contiguously before fitting the tokenizer. The tokenizer is trained on the train
segment only. Validation selects the best checkpoint; the test segment is evaluated after training.
OOV counts, split token counts, source SHA-256, and tokenizer fingerprint are written to `run.json`.

Contiguous code-point splitting can cut a semantic sentence at a boundary. This is acceptable for
the smoke fixture but not necessarily for document-level datasets. A future data pipeline must split
by the true independent document/source unit and record licenses per source.
