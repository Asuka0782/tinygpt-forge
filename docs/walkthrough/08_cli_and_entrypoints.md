# Line-by-line: CLI and package entry points

The CLI is deliberately thin. It turns strings and paths into validated library objects, delegates
the real work, and prints small machine-readable summaries. Model math, checkpoint semantics,
benchmark methodology, and provider security remain independently testable modules.

## `cli.py`

| Source lines | Explanation |
|---|---|
| [L1–L26](../../src/tinygpt_forge/cli.py#L1) | Imports cover argument/JSON/error output, immutable config replacement, paths, PyTorch, and each library entry point. Importing the CLI does not read environment secrets, touch files, allocate a model, or contact a service. |
| [L27–L35](../../src/tinygpt_forge/cli.py#L27) | `_smoke` seeds CPU randomness, loads a validated model TOML, constructs random `[B=2,T=min(16,max_seq_len)]` input/target IDs, and instantiates the decoder. This checks mechanisms, not trained quality. |
| [L36–L43](../../src/tinygpt_forge/cli.py#L36) | Eval/no-grad runs manual and SDPA with the same weights/input, computes max absolute logit error, and fails if it is non-finite or above FP32 tolerance `5e-5`. `status: ok` can no longer accompany a failed equivalence check. |
| [L44–L49](../../src/tinygpt_forge/cli.py#L44) | Train mode restores dropout semantics, computes configured-backend shifted cross-entropy, requires a loss, and runs backward. Thus smoke covers logits, objective, autograd, and parameter gradients without an optimizer step. |
| [L50–L63](../../src/tinygpt_forge/cli.py#L50) | A JSON result exposes seed/runtime/config, parameter count, tensor shapes `[2,T]` and `[2,T,V]`, equivalence error, and loss. It prints UTF-8-friendly formatted JSON and returns process code zero. |
| [L64–L73](../../src/tinygpt_forge/cli.py#L64) | Parser construction has no operational side effects. A required subcommand prevents accidental default work; `smoke` requires config and accepts an explicit seed. |
| [L74–L80](../../src/tinygpt_forge/cli.py#L74) | `train-char` requires config, corpus, and run directory, with optional safe resume and target-step override. The override changes duration only; training later checks the full resume contract. |
| [L81–L92](../../src/tinygpt_forge/cli.py#L81) | `generate-char` requires safe checkpoint/tokenizer/prompt and exposes bounded generation config fields plus CPU/CUDA choice. Static versus dynamic cache is not surfaced yet; the stable baseline uses cached generation internally. |
| [L93–L108](../../src/tinygpt_forge/cli.py#L93) | `benchmark` requires model config/output and exposes device, supported dtype, `[B,T]`, decode length, warm-up, repeats, and seed. Semantic bounds are enforced by `run_benchmark`, not duplicated in argparse. |
| [L109–L122](../../src/tinygpt_forge/cli.py#L109) | `external-chat` exposes prompt/output controls and an intentionally conspicuous cost acknowledgement. Endpoint/model/key/timeout/retry/proxy remain namespaced environment configuration, keeping secrets out of command history. |
| [L123–L130](../../src/tinygpt_forge/cli.py#L123) | `main` parses either real process arguments or an injected sequence used by tests. Smoke dispatch delegates directly and propagates failures as nonzero process termination through `__main__`. |
| [L131–L150](../../src/tinygpt_forge/cli.py#L131) | Training loads model and training sections, optionally creates a revalidated config with a new `max_steps`, and delegates the whole experiment. Printed JSON contains selection/test summaries and run-manifest location, not weights, corpus content, or secrets. |
| [L151–L174](../../src/tinygpt_forge/cli.py#L151) | Generation loads safetensors+JSON onto the chosen device and the saved tokenizer, checks vocabulary identity, encodes prompt to `[1,T] long`, constructs sampling config, runs cached generation, decodes the first sequence, and prints text. Unknown prompt characters fail according to tokenizer rules rather than silently remapping. |
| [L175–L187](../../src/tinygpt_forge/cli.py#L175) | Benchmark dispatch loads config, forwards every measurement parameter, then atomically saves the complete raw evidence before printing a summary. An exception leaves no successful-looking final JSON. |
| [L188–L206](../../src/tinygpt_forge/cli.py#L188) | Console benchmark output keeps artifact path, all correctness gates, and five p50 speedups. Detailed raw samples, environment, memory, throughput, and limitations remain in the saved document to avoid an unreadable terminal dump. |
| [L207–L222](../../src/tinygpt_forge/cli.py#L207) | External work is refused with code 2 before configuration/environment/network access unless the cost flag is present. Only then is the validated provider created and called. The last raise is defensive because argparse restricts command names. Provider failures stay sanitized but are not swallowed. |

## `tinygpt_forge/__init__.py`

| Source lines | Explanation |
|---|---|
| [L1–L5](../../src/tinygpt_forge/__init__.py#L1) | The top-level package imports only the main configuration, output record, and decoder class. Users can write `from tinygpt_forge import TinyGPT` without pulling optional provider names into the public root. |
| [L6–L7](../../src/tinygpt_forge/__init__.py#L6) | `__all__` declares the stable root surface. The source version mirrors `pyproject.toml`; release procedure and tests must update/check both until version metadata is centralized. |

## `tinygpt_forge/__main__.py`

| Source lines | Explanation |
|---|---|
| [L1–L4](../../src/tinygpt_forge/__main__.py#L1) | This tiny adapter imports `main`, enabling `python -m tinygpt_forge ...` in addition to the installed `tinygpt` console script. Import alone still performs no command. |
| [L5–L6](../../src/tinygpt_forge/__main__.py#L5) | The standard module guard calls `main` only during execution and raises `SystemExit` with its integer return code, preserving shell success/refusal semantics. |

## `model/__init__.py`

| Source lines | Explanation |
|---|---|
| [L1–L4](../../src/tinygpt_forge/model/__init__.py#L1) | The model subpackage re-exports the key attention class, complete model, and structured output while implementation helpers remain reachable from their own modules. |
| [L5–L6](../../src/tinygpt_forge/model/__init__.py#L5) | `__all__` documents the intended subpackage API and prevents incidental imported names from becoming an accidental compatibility promise. |

## Execution boundary

`argv → argparse namespace → validated library contracts → stdout/stderr + exit code`.
Keeping this layer small lets notebooks, tests, future servers, and the CLI share the same model and
experiment semantics instead of growing separate implementations.
