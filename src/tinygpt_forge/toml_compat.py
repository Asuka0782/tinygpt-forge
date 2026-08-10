"""Typed TOML loading across Python 3.10 and 3.11+."""

from __future__ import annotations

import importlib
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Protocol, cast

from tinygpt_forge.serialization import read_bounded_bytes


class _TomlModule(Protocol):
    @staticmethod
    def load(file: BinaryIO, /) -> object: ...


def load_toml(path: str | Path) -> dict[str, Any]:
    """Load a TOML document using stdlib `tomllib` or the Python 3.10 fallback."""

    module_name = "tomllib" if sys.version_info >= (3, 11) else "tomli"
    module = cast(_TomlModule, importlib.import_module(module_name))
    payload = read_bounded_bytes(path, kind="TOML configuration")
    document = module.load(BytesIO(payload))
    if not isinstance(document, dict) or not all(isinstance(key, str) for key in document):
        raise ValueError("TOML root must be a string-keyed table")
    return cast(dict[str, Any], document)
