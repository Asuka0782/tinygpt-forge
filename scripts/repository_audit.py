"""Fail on common public-repository hygiene and secret-leak mistakes."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

MAX_PUBLIC_FILE_BYTES = 5 * 1024 * 1024
FORBIDDEN_DIRECTORY_NAMES = {
    "." + "co" + "dex",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    ".vscode",
    "__pycache__",
    "build",
    "checkpoints",
    "dist",
    "mlruns",
    "outputs",
    "runs",
    "venv",
    "wandb",
}
FORBIDDEN_FILE_NAMES = {".env", ".env.local", ".env.production"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".pth", ".pt", ".bin"}
TEXT_SUFFIXES = {
    ".cff",
    ".css",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

CONTENT_PATTERNS = {
    "private Windows user path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.IGNORECASE),
    # Split the sentinels so the audit implementation does not flag its own source.
    "private username": re.compile(r"(?<!\d)" + "175" + r"29(?!\d)"),
    "assistant work trace": re.compile(
        r"\." + "co" + "dex|" + "pasted" + "-text|" + "agent work" + " file",
        re.IGNORECASE,
    ),
    "OpenAI-style secret": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    "Hugging Face secret": re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"),
    "GitHub personal token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
}


def _iter_public_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        relative_parts = path.relative_to(root).parts
        if any(
            part == ".git" or part in FORBIDDEN_DIRECTORY_NAMES or part.endswith(".egg-info")
            for part in relative_parts
        ):
            continue
        if path.is_file():
            files.append(path)
    return sorted(files)


def audit_repository(root: Path, *, release: bool) -> dict[str, Any]:
    """Return a machine-readable audit; callers decide where to save it."""

    issues: list[dict[str, str]] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_dir()):
        relative_parts = path.relative_to(root).parts
        if ".git" in relative_parts:
            continue
        if path.name in FORBIDDEN_DIRECTORY_NAMES or path.name.endswith(".egg-info"):
            issues.append(
                {
                    "kind": "forbidden generated/private directory",
                    "path": path.relative_to(root).as_posix(),
                }
            )

    files = _iter_public_files(root)
    for path in files:
        relative = path.relative_to(root).as_posix()
        if path.name in FORBIDDEN_FILE_NAMES:
            issues.append({"kind": "forbidden file", "path": relative})
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.append({"kind": "forbidden artifact suffix", "path": relative})
        size = path.stat().st_size
        if size > MAX_PUBLIC_FILE_BYTES:
            issues.append(
                {
                    "kind": "large file",
                    "path": relative,
                    "detail": f"{size} bytes exceeds {MAX_PUBLIC_FILE_BYTES}",
                }
            )
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {".gitignore"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            issues.append({"kind": "non-UTF-8 public text", "path": relative})
            continue
        for label, pattern in CONTENT_PATTERNS.items():
            match = pattern.search(text)
            if match is not None:
                issues.append(
                    {
                        "kind": label,
                        "path": relative,
                        "detail": f"match at character {match.start()}",
                    }
                )

    required = {
        "README.md",
        "README_zh.md",
        "pyproject.toml",
        ".gitignore",
        ".gitattributes",
        ".env.example",
        "CONTRIBUTING.md",
        "SECURITY.md",
        ".github/workflows/ci.yml",
    }
    if release:
        required.add("LICENSE")
    for relative in sorted(required):
        if not (root / relative).is_file():
            issues.append({"kind": "missing required file", "path": relative})

    return {
        "format": "tinygpt-forge-repository-audit-v1",
        "mode": "release" if release else "draft",
        "root_name": root.name,
        "files_scanned": len(files),
        "max_public_file_bytes": MAX_PUBLIC_FILE_BYTES,
        "status": "passed" if not issues else "failed",
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    document = audit_repository(args.root.resolve(), release=args.release)
    rendered = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0 if document["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
