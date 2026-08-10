# Line-by-line: bounded strict metadata parsing

Safetensors removes pickle execution, but its JSON/TOML companions still need resource and
ambiguity boundaries. This module is intentionally independent of PyTorch so configs, tokenizers,
and checkpoints share one small parser contract.

## `serialization.py`

| Source lines | Explanation |
|---|---|
| [L1–L10](../../src/tinygpt_forge/serialization.py#L1) | Imports provide JSON, paths, and typed dynamic values. `MAX_METADATA_BYTES=2 MiB` applies to small manifests/configuration—not tensor weights or corpora. |
| [L11–L18](../../src/tinygpt_forge/serialization.py#L11) | `read_bounded_bytes` requires an explicit human-readable `kind` for safe errors and permits a smaller caller-supplied positive limit. |
| [L19–L26](../../src/tinygpt_forge/serialization.py#L19) | Reading `limit+1` bytes detects overflow without loading the rest of an adversarial file. Exact-limit payloads pass; oversized metadata fails before decoding or parser allocation. This is O(limit) memory. |
| [L27–L35](../../src/tinygpt_forge/serialization.py#L27) | JSON's `object_pairs_hook` sees keys before a dict would overwrite duplicates. Repeated keys fail, avoiding ambiguous manifests where different parsers choose first versus last values. The check applies recursively. |
| [L36–L40](../../src/tinygpt_forge/serialization.py#L36) | Python's JSON parser normally accepts non-standard `NaN`/`Infinity`; `parse_constant` routes them here and rejects them, keeping numeric metadata finite and portable. |
| [L41–L57](../../src/tinygpt_forge/serialization.py#L41) | Bytes decode strictly as UTF-8, parse with unique-object/nonfinite hooks, and must produce an object root. Decode/syntax/policy failures become a stable message with the original cause chained. The final cast follows the runtime dict check; it performs no conversion. |

## Boundary

This protects parser memory and manifest interpretation. It does not authenticate files, constrain
safetensors dimensions, or sandbox a model; SHA-256/source trust and isolated resource limits remain
separate requirements.
