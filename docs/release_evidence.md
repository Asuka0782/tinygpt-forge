# Release and CI evidence

Evidence date: 2026-08-10

## Public repository state

- Repository: [Asuka0782/tinygpt-forge](https://github.com/Asuka0782/tinygpt-forge)
- Visibility: public
- Default branch: `main`
- Evidence commit: `61412c8b49c2d2cdba3f07ec331e22d7acbcb854`
- Project license: Apache-2.0, recognized by GitHub and included in wheel metadata
- Repository history at this snapshot: 10 scoped commits

This is a verified public source snapshot, not a stable API promise. No GitHub Release, Git tag,
PyPI upload, signed provenance attestation, or long-term support policy is claimed.

## Successful GitHub Actions run

The [successful CI run](https://github.com/Asuka0782/tinygpt-forge/actions/runs/31381403820)
checked the exact evidence commit above.

| Job | Pinned environment | Evidence produced |
|---|---|---|
| Ruff | GitHub Ubuntu runner | lint and format checks passed |
| CPU tests | Python 3.10, PyTorch 2.3.0 CPU | repository hygiene, 57 tests, installed CLI smoke |
| CPU tests | Python 3.14, PyTorch 2.13.0 CPU | repository hygiene, 57 tests, installed CLI smoke |
| Mypy | Python 3.10, PyTorch 2.3.0 CPU | strict type check passed for 22 source files |
| Build wheel | Python 3.12, PyTorch 2.13.0 CPU | sdist/wheel build and non-editable wheel smoke passed |

The two endpoint jobs provide direct evidence for those combinations only. They do not prove every
intermediate dependency resolver result, operating system, accelerator, or future package release.

## Failures retained as engineering evidence

The first public runs were not hidden or rewritten:

1. A clean runner exposed NumPy as a real safetensors checkpoint dependency; local NumPy had masked
   it. Runtime markers now select NumPy 1.x for Python below 3.13 and NumPy 2.x for newer Python.
2. Type-checking Python-3.10-targeted code against Python 3.14 NumPy stubs produced a stub-language
   mismatch. Mypy now runs in the declared minimum environment instead of suppressing the import.
3. PyTorch 2.3 lacks the later `is_flash_attention_available` helper. The benchmark now capability-
   probes that optional API and safely records `false` when absent rather than raising.

The subsequent run passed without removing the Python 3.10/PyTorch 2.3 lower-bound job.

## Local package and hardware evidence

The source snapshot produced a 46,288-byte wheel with SHA-256
`012e5065a8da27054314d63184a34fb9c85fa4b8d8fba460f37bd68d8ee91749`. It was installed outside
the repository and reported:

- `License-Expression: Apache-2.0` and packaged `licenses/LICENSE`;
- Python requirement `>=3.10,<3.15`;
- PyTorch requirement `>=2.3,<3.0`;
- version-conditioned NumPy requirements;
- a successful `python -m tinygpt_forge smoke` using Python 3.14.3 and PyTorch 2.11.0+cu128.

GPU training and benchmark evidence is limited to the recorded RTX 5060 Laptop GPU artifacts in
`results/`. See [benchmark methodology](10_benchmarks.md) for shapes, raw samples, negative results,
and interpretation boundaries.

## Hygiene evidence

The final release-mode repository audit scanned 98 public files and reported no forbidden generated
directories, private paths, username, common secret patterns, collaboration traces, or files above
5 MiB. Git history was checked with `git fsck --full`, the working tree was clean, and a tracked-
content sensitive-pattern scan had no matches. These checks reduce accidental exposure; they are
not a proof that arbitrary future commits are safe.
