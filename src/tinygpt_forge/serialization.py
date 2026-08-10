"""Bounded, strict readers for small public metadata files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

MAX_METADATA_BYTES = 2 * 1024 * 1024


def read_bounded_bytes(
    path: str | Path,
    *,
    kind: str,
    max_bytes: int = MAX_METADATA_BYTES,
) -> bytes:
    """Read at most ``max_bytes`` and fail before parsing oversized metadata."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    with Path(path).open("rb") as handle:
        payload = handle.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(f"{kind} exceeds the {max_bytes}-byte metadata limit")
    return payload


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError("duplicate JSON object key")
        document[key] = value
    return document


def _reject_nonfinite_constant(value: str) -> None:
    del value
    raise ValueError("non-finite JSON number")


def read_json_object(path: str | Path, *, kind: str) -> dict[str, Any]:
    """Read one bounded UTF-8 JSON object with unique keys and finite numbers."""

    payload = read_bounded_bytes(path, kind=kind)
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{kind} must be strict UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{kind} must contain a JSON object")
    return cast(dict[str, Any], document)
