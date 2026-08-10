# Source walkthrough (v0.0.1 draft)

This walkthrough follows the current source layout after the first API stabilization pass. Line
references must be revalidated before release if code changes.

## End-to-end execution path

```text
TOML
  -> ModelConfig / TrainingConfig validation
  -> raw-text split -> CharacterTokenizer(train only) -> token IDs
  -> NextTokenBatcher -> input/target [B,T]
  -> TinyGPT.forward
       embedding
       N × DecoderBlock
         RMSNorm -> CausalSelfAttention -> residual
         RMSNorm -> SwiGLU -> residual
       final RMSNorm -> tied LM head -> logits [B,T,V]
  -> aligned cross-entropy -> backward -> AdamW
  -> safetensors + JSON checkpoint
```

## `config.py`

[`ModelConfig`](../src/tinygpt_forge/config.py#L16) is frozen so runtime code cannot silently mutate
the semantic model definition.

- Lines 18—29 declare vocabulary/context/model widths, Q/KV head counts, RoPE/RMSNorm values,
  dropout, weight tying, and backend.
- [`__post_init__`](../src/tinygpt_forge/config.py#L32) rejects non-positive dimensions,
  `d_model % n_heads != 0`, incompatible GQA grouping, odd RoPE head dimensions, invalid dropout,
  and unknown backend names.
- [`head_dim`](../src/tinygpt_forge/config.py#L62) computes `C/Hq`; [`query_groups`](../src/tinygpt_forge/config.py#L68)
  computes `Hq/Hkv`.
- [`from_mapping`](../src/tinygpt_forge/config.py#L79) rejects unknown TOML keys instead of ignoring a
  typo that could silently change an experiment.
- [`from_toml`](../src/tinygpt_forge/config.py#L90) requires a `[model]` table and delegates parsing to
  the typed Python 3.10/3.11 compatibility wrapper.

A more permissive alternative would use `raw.get(name, default)` everywhere. It is shorter but can
turn `n_head=8` into an ignored typo while the default `n_heads` runs, invalidating comparisons.

## `model/rope.py`

[`RotaryEmbedding`](../src/tinygpt_forge/model/rope.py#L9) precomputes inverse frequencies, not a
full position table.

- The constructor validates even `head_dim` and positive base, creates indices `0,2,...,D-2`, and
  registers `inv_freq` as a non-persistent buffer. Non-persistent means it follows device moves but
  is recomputable and omitted from checkpoints.
- [`forward`](../src/tinygpt_forge/model/rope.py#L24) checks `[B,H,T,D]`, constructs or validates the
  `T` absolute positions, computes angles in FP64 for FP64 input and FP32 otherwise, then casts
  sin/cos to activation dtype.
- Slices `0::2` and `1::2` form adjacent pairs. The two rotation equations are stacked and flattened
  back to `D`.

The simpler alternative is a Python loop over positions/pairs; it matches the formula but prevents
efficient vectorized GPU execution.

## `model/components.py`

[`RMSNorm`](../src/tinygpt_forge/model/components.py#L12):

- creates one learned scale vector initialized to one;
- promotes FP16/BF16 input to FP32 for `mean(x²)` and reciprocal square root;
- casts normalized values back, then applies learned scale.

[`SwiGLU`](../src/tinygpt_forge/model/components.py#L33) creates bias-free gate/up/down projections.
Its [`forward`](../src/tinygpt_forge/model/components.py#L42) computes
`down(silu(gate(x)) * up(x))`; all leading `[B,T]` dimensions are preserved.

## `model/attention.py`

[`AttentionOutput`](../src/tinygpt_forge/model/attention.py#L20) names the three possible products:
hidden states, observable manual probabilities, and Dynamic Cache K/V.

[`repeat_key_value`](../src/tinygpt_forge/model/attention.py#L28) expands Hkv to Hq only for attention
math. Returning the input unchanged for one group avoids an unnecessary allocation in MHA.

[`CausalSelfAttention.__init__`](../src/tinygpt_forge/model/attention.py#L43):

- Q output width is `Hq×Dh=C`;
- K/V output width is `Hkv×Dh`, which is smaller for GQA/MQA;
- output projection returns to `C`;
- RoPE belongs to attention because it transforms Q/K, not the residual stream.

[`_project`](../src/tinygpt_forge/model/attention.py#L53) performs `[B,T,C] -> [B,H,T,Dh]`. `transpose`
makes heads explicit. Q/K receive RoPE; V does not.

[`_manual_attention`](../src/tinygpt_forge/model/attention.py#L74):

1. QK transpose produces `[B,Hq,Q,K]` scores.
2. `1/sqrt(Dh)` scales them.
3. absolute query/key ranges build `key_position <= query_position`.
4. disallowed scores become `-inf` before softmax.
5. dropout applies to probabilities during training.
6. probability-value multiplication returns `[B,Hq,Q,Dh]`.

[`forward`](../src/tinygpt_forge/model/attention.py#L100) is the key branch point. It validates
backend/cache exclusivity, derives past length, validates Dynamic Cache shape, creates absolute RoPE
positions, appends or writes new K/V, expands groups, and selects manual or SDPA. Square no-past
SDPA uses `is_causal=True`; cached non-square SDPA receives an explicit mask. Finally heads are
transposed/contiguous/viewed to `[B,T,C]` and projected.

Likely bugs this structure prevents: caching repeated Hq K/V, rotating V, applying RoPE position zero
to every decode step, and using the wrong non-square causal alignment.

## `model/gpt.py`

[`CausalLMOutput`](../src/tinygpt_forge/model/gpt.py#L18) keeps logits mandatory and makes loss,
attention maps, and Dynamic Cache optional.

[`DecoderBlock`](../src/tinygpt_forge/model/gpt.py#L27) constructs two RMSNorms and two residual
sublayers. [`forward`](../src/tinygpt_forge/model/gpt.py#L38) normalizes before each sublayer and
returns cache/observability products without storing them on the module.

[`TinyGPT.__init__`](../src/tinygpt_forge/model/gpt.py#L68):

- creates token embedding, dropout, `n_layers` blocks, final RMSNorm, and LM head;
- initializes Linear/Embedding weights with normal std 0.02;
- assigns `lm_head.weight = token_embedding.weight` after initialization to share one Parameter.

[`TinyGPT.forward`](../src/tinygpt_forge/model/gpt.py#L85):

1. validates rank, integer dtype, non-empty input, and context capacity;
2. rejects mixing Dynamic and Static cache APIs;
3. checks per-layer cache count/length and Static Cache model/batch match;
4. embeds `[B,T] -> [B,T,C]`;
5. loops blocks, collecting manual attention or Dynamic Cache only when requested;
6. commits Static Cache length after all layers succeed;
7. applies final RMSNorm and tied head;
8. computes aligned loss only if targets were supplied.

[`parameter_count`](../src/tinygpt_forge/model/gpt.py#L186) iterates unique Parameter objects, so tied
embedding/head storage is counted once.

## `cache.py`

[`StaticKVCache`](../src/tinygpt_forge/cache.py#L11) allocates K and V tensors for every layer.

- Constructor validates batch/capacity/dtype and normalizes `cuda` to the actual allocated device
  such as `cuda:0`.
- [`update`](../src/tinygpt_forge/cache.py#L49) validates layer, logical start, shape, device, and dtype;
  it copies new K/V into `[start:end]` and returns prefix views.
- [`advance`](../src/tinygpt_forge/cache.py#L89) commits one completed model call.
- [`reset`](../src/tinygpt_forge/cache.py#L98) discards the logical sequence without clearing memory.
- [`rewind`](../src/tinygpt_forge/cache.py#L103) retains a prefix for repeated steady-state tests.
- [`allocated_bytes`](../src/tinygpt_forge/cache.py#L111) calculates real tensor storage, not a
  theoretical Hq-expanded cache.

The class is mutable by design; ownership stays inside one generation request. Sharing it across
concurrent requests would corrupt logical positions and is unsupported.

## `losses.py`, `data.py`, and `tokenizer.py`

[`aligned_next_token_cross_entropy`](../src/tinygpt_forge/losses.py#L9) validates `[B,T,V]` versus
`[B,T]`, flattens, and calls cross-entropy. [`shifted_next_token_cross_entropy`](../src/tinygpt_forge/losses.py#L37)
drops final logits/first labels before delegating.

[`split_text`](../src/tinygpt_forge/data.py#L62) splits raw text before vocabulary fitting.
[`split_token_ids`](../src/tinygpt_forge/data.py#L86) validates fractions and minimum next-token
length. [`NextTokenBatcher`](../src/tinygpt_forge/data.py#L109) stores IDs on CPU, samples start indices
with its own generator, builds shifted windows, and transfers completed batches to the device.

[`CharacterTokenizer`](../src/tinygpt_forge/tokenizer.py#L18) enforces unique one-code-point entries
plus `<unk>`. Training sorts unique code points. Save/load use a versioned UTF-8 JSON object, and
fingerprinting hashes canonical JSON rather than a local path.

## `generation.py`

[`GenerationConfig`](../src/tinygpt_forge/generation.py#L16) validates output count, temperature, and
top-k. [`sample_next_token`](../src/tinygpt_forge/generation.py#L33) uses argmax at zero temperature;
otherwise it scales, applies top-k threshold, softmaxes, and samples with a provided generator.

[`generate`](../src/tinygpt_forge/generation.py#L53) validates context bounds, creates a device-local
generator, optionally allocates Static Cache, and loops exactly `max_new_tokens`. Dynamic mode feeds
returned tuples forward; Static mode mutates owned storage; full mode feeds the entire growing
sequence. All paths share sampling code.

## `checkpoint.py`

[`save_weights`](../src/tinygpt_forge/checkpoint.py#L32) writes model-aware safetensors to a temporary
file, atomically replaces the target, hashes it, then atomically writes JSON. [`load_weights`](../src/tinygpt_forge/checkpoint.py#L63)
validates format/local filename/hash/config and performs strict model loading.

Optimizer serialization separates tensor state into safetensors and tuple/scalar structure into a
tagged JSON codec. RNG serialization stores global CPU, available CUDA, and batcher states.

[`save_training_checkpoint`](../src/tinygpt_forge/checkpoint.py#L257) writes model, optimizer, RNG,
GradScaler, and trainer state; [`load_training_checkpoint`](../src/tinygpt_forge/checkpoint.py#L291)
validates hashes/config before restoring them. No public checkpoint path calls `torch.load`.

## `training.py`

[`TrainingConfig`](../src/tinygpt_forge/training.py#L34) validates optimizer, split, device, precision,
and loop controls. [`resume_contract`](../src/tinygpt_forge/training.py#L82) excludes only
`max_steps`, so a resumed run may extend duration without silently changing learning rate/data shape.

[`evaluate`](../src/tinygpt_forge/training.py#L127) resets a fixed validation sampler state, switches
to eval/inference mode, applies the same dtype/backend, averages losses, and restores training mode.

[`train_character_model`](../src/tinygpt_forge/training.py#L170) is the orchestration path:

- refuses accidental overwrite unless resume was explicit;
- hashes/splits text, fits or verifies tokenizer, checks split lengths;
- resolves device/vocabulary, seeds model, constructs AdamW/scaler/batchers;
- optionally restores the complete checkpoint and validates source/tokenizer/config;
- evaluates step zero only for a new run;
- accumulates microbatch gradients, clips, steps, evaluates, saves best/resume;
- saves last weights, evaluates test, and atomically writes the run manifest.

## `benchmark.py` and `cli.py`

[`_measure`](../src/tinygpt_forge/benchmark.py#L52) warms up, synchronizes CUDA, resets peak memory,
times every sample with `perf_counter_ns`, synchronizes again, and retains raw milliseconds.
[`run_benchmark`](../src/tinygpt_forge/benchmark.py#L94) executes correctness gates before manual/
SDPA and full/Dynamic/Static timings, including isolated single-token decode.

[`build_parser`](../src/tinygpt_forge/cli.py#L66) defines explicit commands and bounded arguments.
[`main`](../src/tinygpt_forge/cli.py#L125) dispatches without importing paid provider configuration
for local commands. External chat requires a cost acknowledgement before constructing the provider.

## Package boundaries: `__init__.py`, `__main__.py`, and `toml_compat.py`

The root [`__all__`](../src/tinygpt_forge/__init__.py#L6) exposes only the stable learning surface:
configuration, model output, and model. The version is a plain string on the following line; it must
match `pyproject.toml` before a release. `model/__init__.py` and `providers/__init__.py` repeat this
pattern so internal helpers do not become accidental public APIs.

The [module guard](../src/tinygpt_forge/__main__.py#L5) raises `SystemExit(main())`. Returning an
integer from `main` is not enough when executing `python -m tinygpt_forge`; raising converts that
integer into the process exit code, which is why a refused paid request exits with code 2.

[`_TomlModule`](../src/tinygpt_forge/toml_compat.py#L14) is a tiny structural type shared by the
stdlib `tomllib` and Python 3.10's `tomli`. [`load_toml`](../src/tinygpt_forge/toml_compat.py#L19)
chooses the module by interpreter version, bounds bytes before either loader parses them, and
validates a string-keyed root table. The cast informs static analysis; it does not convert data.

## `serialization.py`

[`read_bounded_bytes`](../src/tinygpt_forge/serialization.py#L12) reads one sentinel byte beyond a
2 MiB metadata limit and rejects overflow before parsing. [`read_json_object`](../src/tinygpt_forge/serialization.py#L43)
then requires strict UTF-8, unique keys at every object level, finite JSON numbers, and an object
root. Checkpoints and Tokenizers share this contract; tensor/corpus payloads use separate paths.

## `providers/base.py` and `providers/openai_compatible.py`

[`ProviderError`](../src/tinygpt_forge/providers/base.py#L10) is the only public operational failure
type and is documented to contain sanitized text. [`ChatMessage`](../src/tinygpt_forge/providers/base.py#L15)
validates the role, non-empty content, and per-message length before networking.
[`ChatProvider`](../src/tinygpt_forge/providers/base.py#L30) is a `Protocol`: callers can depend on
the shape of `complete` without importing a specific vendor client.

[`TransportResponse`](../src/tinygpt_forge/providers/openai_compatible.py#L22) deliberately contains
only status and bytes. [`HTTPTransport`](../src/tinygpt_forge/providers/openai_compatible.py#L49)
makes the network edge injectable; tests pass a fake, so CI cannot accidentally reach a paid
service. [`UrllibTransport.send`](../src/tinygpt_forge/providers/openai_compatible.py#L65) builds an
opener with only the explicitly configured proxy and a no-redirect handler, applies a bounded
timeout, reads at most 2 MiB plus one sentinel byte, converts HTTP errors into status/body data,
and replaces connection details with a credential-free internal error. Refusing redirects prevents
the bearer header from crossing the configured endpoint boundary.

[`_validated_url`](../src/tinygpt_forge/providers/openai_compatible.py#L93) requires an absolute URL,
rejects embedded credentials/query/fragment, and allows plain HTTP only on loopback hosts.
[`OpenAICompatibleConfig`](../src/tinygpt_forge/providers/openai_compatible.py#L108) hides the key from
`repr`, rejects header control characters, and bounds timeouts/retries. Its
[`from_environment`](../src/tinygpt_forge/providers/openai_compatible.py#L146) reads only namespaced
variables; it never dumps the process environment or silently parses a local file.

[`OpenAICompatibleProvider.complete`](../src/tinygpt_forge/providers/openai_compatible.py#L182)
validates aggregate input/output controls, constructs one JSON request, retries only timeout/rate/
server statuses with bounded exponential backoff, rejects oversized bodies, and never copies a
provider body into an exception. [`_parse_content`](../src/tinygpt_forge/providers/openai_compatible.py#L250)
accepts only the minimal chat-completions shape and a non-empty string.

The simpler alternative is to call `urllib.request.urlopen` directly from the CLI. That couples
parsing, secrets, retries, and user interaction, and makes an offline CI guarantee much harder to
prove. The current boundary costs a few small types but keeps the local model path independent.

## Maintenance warning

Line anchors are documentation data. After source edits, compare every linked symbol/line and update
this file before release. The documentation test verifies that each backticked Python symbol still
appears on its linked line, so a refactor fails CI instead of silently leaving stale teaching notes.
