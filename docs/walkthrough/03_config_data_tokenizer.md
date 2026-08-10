# Line-by-line: configuration, data, tokenizer, and TOML

These modules sit before the neural network. Their main correctness job is preventing a typo,
split leak, vocabulary drift, or RNG drift from silently changing the experiment.

## `config.py`

| Source lines | Explanation |
|---|---|
| [L1–L14](../../src/tinygpt_forge/config.py#L1) | The module imports immutable dataclass/reflection helpers, path/types, and the cross-version TOML loader. `ATTENTION_BACKENDS` is a frozen set, so runtime code cannot mutate the accepted semantic choices. |
| [L15–L18](../../src/tinygpt_forge/config.py#L15) | A frozen `ModelConfig` is a value object: cache/model equality is structural, and code cannot silently mutate head counts midway through a run. |
| [L19–L30](../../src/tinygpt_forge/config.py#L19) | Required fields define vocabulary/context, residual width, layer count, Q/KV heads, and FFN width. Defaults define RoPE base, RMS epsilon, dropout, embedding tying, and SDPA. These are semantic values shared by both backends; no performance-only switch changes weights. |
| [L31–L44](../../src/tinygpt_forge/config.py#L31) | `__post_init__` builds named dimension/value pairs and rejects bools despite `bool` being a Python subclass of `int`. Checking all positive integers once produces precise field errors before tensors allocate. |
| [L45–L59](../../src/tinygpt_forge/config.py#L45) | Divisibility makes `Dh=C/Hq` integral and GQA grouping `Hq/Hkv` exact. Even `Dh` is required by pairwise RoPE. Positive RoPE/RMS constants, dropout in `[0,1)`, and a closed backend set prevent invalid numeric branches. |
| [L60–L66](../../src/tinygpt_forge/config.py#L60) | `head_dim` is derived instead of separately configured, removing one inconsistent degree of freedom. Integer division is safe because validation already proved divisibility. |
| [L67–L72](../../src/tinygpt_forge/config.py#L67) | `query_groups=Hq/Hkv`: `1` is MHA, `Hq` with `Hkv=1` is MQA, intermediate values are GQA. |
| [L73–L77](../../src/tinygpt_forge/config.py#L73) | `asdict` returns a fresh JSON-safe mapping. Checkpoints compare this complete contract, rather than a hand-maintained subset that could omit a new field. |
| [L78–L88](../../src/tinygpt_forge/config.py#L78) | `from_mapping` reflects dataclass field names, rejects unknown keys, then invokes the normal constructor and all validation. Rejecting `n_head` instead of ignoring it is critical for reproducible config-driven experiments. |
| [L89–L98](../../src/tinygpt_forge/config.py#L89) | `from_toml` normalizes the path, loads a document, requires a `[model]` mapping, and delegates to one mapping path. File parsing and semantic validation therefore cannot disagree. |

## `data.py`

| Source lines | Explanation |
|---|---|
| [L1–L11](../../src/tinygpt_forge/data.py#L1) | The boundary owns leakage-conscious contiguous splitting and a deterministic sampler. `Sequence` accepts lists/tuples, while tensor batches use PyTorch's explicit `Tensor` type. |
| [L12–L20](../../src/tinygpt_forge/data.py#L12) | `TokenSplits` freezes three rank-one ID tensors into named roles, preventing positional tuple mix-ups between validation and test. |
| [L21–L29](../../src/tinygpt_forge/data.py#L21) | `TextSplits` performs the same role before tokenization. This is the object used by training so the vocabulary can be fitted on train text only. |
| [L30–L41](../../src/tinygpt_forge/data.py#L30) | `_split_counts` centralizes fraction/count validation using integers only. Each fraction lies in `[0,1)`, their sum leaves a training region, and at least one next-token pair is possible. It avoids the old wasteful approach of constructing `range(N)` and an `int64` tensor merely to calculate lengths. |
| [L42–L59](../../src/tinygpt_forge/data.py#L42) | Validation/test counts use floor via `int`; training receives the remainder, so counts sum exactly to `N`. Any requested nonzero split needs at least two items for next-token scoring. The function returns three integers with (O(1)) memory/time. |
| [L60–L68](../../src/tinygpt_forge/data.py#L60) | `split_text` accepts raw Unicode text and keyword-only fractions. Its contract explicitly precedes vocabulary fitting. |
| [L69–L83](../../src/tinygpt_forge/data.py#L69) | Empty text is rejected. Count validation runs on Python string length, then slice boundaries create contiguous train/validation/test substrings. Contiguous splits preserve temporal order but can expose distribution shift; that is intentional and documented, not random leakage. |
| [L84–L92](../../src/tinygpt_forge/data.py#L84) | `split_token_ids` is retained for already-tokenized experiments and applies the same count policy. It must not be used to fit preprocessing statistics on all three splits afterward. |
| [L93–L106](../../src/tinygpt_forge/data.py#L93) | Counts are validated first, IDs become one `torch.long` tensor, and three slices are cloned. Clones prevent later in-place modification of one split from aliasing shared backing storage. |
| [L107–L120](../../src/tinygpt_forge/data.py#L107) | `NextTokenBatcher` receives one split plus block/batch/seed/device. Sampling happens on CPU from stored CPU IDs, then completed batches transfer to the target device; this keeps its RNG state portable for exact resume. |
| [L121–L132](../../src/tinygpt_forge/data.py#L121) | IDs must be rank-one `long`; block/batch are positive; sequence length must exceed block size because every input needs a following target. The sampler owns a dedicated CPU `Generator`, isolated from dropout/model initialization RNG. |
| [L133–L150](../../src/tinygpt_forge/data.py#L133) | Valid start indices are sampled as `[B]`. Each input slice has `block_size` IDs and each target slice starts one position later, producing aligned `[B,T]` tensors. List comprehension is clear for the tiny baseline; a vectorized gather or memory-mapped loader is a future scale optimization. Transfers occur after stacking to avoid `B` small device copies. |
| [L151–L156](../../src/tinygpt_forge/data.py#L151) | `get_rng_state` clones the generator byte state so later sampling cannot mutate the saved checkpoint tensor by aliasing. |
| [L157–L160](../../src/tinygpt_forge/data.py#L157) | Restore forces state to CPU, matching the generator's device. Exact training continuation depends on installing this state after loading the checkpoint. |

## `tokenizer.py`

| Source lines | Explanation |
|---|---|
| [L1–L16](../../src/tinygpt_forge/tokenizer.py#L1) | Hash/JSON/atomic-replace imports support a deterministic character baseline; loading uses the shared bounded strict JSON reader. `TOKENIZER_FORMAT` versions the schema independently of package version. |
| [L17–L23](../../src/tinygpt_forge/tokenizer.py#L17) | The frozen tokenizer stores an ordered token tuple and unknown-token string. IDs are tuple indices, so order is model semantics and must not mutate. |
| [L24–L33](../../src/tinygpt_forge/tokenizer.py#L24) | Validation requires a nonempty unique vocabulary containing `<unk>`. Every other token is one Python Unicode code point, not a grapheme, byte, BPE, or normalized unit. |
| [L34–L44](../../src/tinygpt_forge/tokenizer.py#L34) | Training rejects empty text, sorts unique code points deterministically, rejects sentinel collision, and places `<unk>` first. Complexity is (O(N+V\log V)). |
| [L45–L56](../../src/tinygpt_forge/tokenizer.py#L45) | `vocab_size` is tuple length. `unk_id` locates the configured sentinel rather than assuming zero, so loaded custom valid orderings work. |
| [L57–L73](../../src/tinygpt_forge/tokenizer.py#L57) | Encode builds a token→ID dictionary once and scans in (O(N)). Strict mode raises at first OOV; non-strict maps to unknown. Train encoding is strict while validation/test record OOV. |
| [L74–L85](../../src/tinygpt_forge/tokenizer.py#L74) | Decode accepts any iterable, rejects bool/non-int/negative/out-of-range IDs, collects pieces, and joins once. `<unk>` expands to multiple characters. |
| [L86–L94](../../src/tinygpt_forge/tokenizer.py#L86) | `to_dict` records schema, unknown token, and ordered list. No local path or corpus content is embedded. |
| [L95–L105](../../src/tinygpt_forge/tokenizer.py#L95) | Fingerprinting canonicalizes UTF-8 JSON and SHA-256 hashes it. Pretty whitespace/location cannot change identity; the hash proves equality, not publisher trust. |
| [L106–L118](../../src/tinygpt_forge/tokenizer.py#L106) | Save writes pretty UTF-8 JSON to a sibling temp and atomically replaces the destination, preserving the previous complete file until publish. |
| [L119–L132](../../src/tinygpt_forge/tokenizer.py#L119) | Load first enforces the 2 MiB, unique-key, finite strict-JSON boundary, checks schema/field types, then reconstructs through dataclass invariants. It never executes pickle or imports file-named classes. |

## `toml_compat.py`

| Source lines | Explanation |
|---|---|
| [L1–L13](../../src/tinygpt_forge/toml_compat.py#L1) | This adapter isolates Python-version variation and imports `BytesIO` plus the shared bounded reader. `tomllib` is standard from 3.11; `tomli` is the conditional 3.10 dependency. |
| [L14–L18](../../src/tinygpt_forge/toml_compat.py#L14) | `_TomlModule` is structural typing: any imported module with static-like `load(BinaryIO)->object` fits. The ellipsis is a protocol stub. |
| [L19–L21](../../src/tinygpt_forge/toml_compat.py#L19) | `load_toml` accepts string/path and promises a string-keyed mapping after validation. |
| [L22–L25](../../src/tinygpt_forge/toml_compat.py#L22) | Version selects stdlib/fallback parser; at most 2 MiB is read before parsing and wrapped as a binary stream. `cast` changes static knowledge, not runtime data. |
| [L26–L28](../../src/tinygpt_forge/toml_compat.py#L26) | Root must be a string-keyed dictionary. Section loaders still validate `[model]`/`[training]` and reject unknown fields. |
