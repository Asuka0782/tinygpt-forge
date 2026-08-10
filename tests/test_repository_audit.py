from __future__ import annotations

import importlib.util
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]


def _load_audit_module() -> ModuleType:
    path = ROOT / "scripts" / "repository_audit.py"
    spec = importlib.util.spec_from_file_location("repository_audit_for_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load repository audit module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RepositoryAuditTests(unittest.TestCase):
    def test_generated_directory_is_reported_instead_of_silently_skipped(self) -> None:
        audit_repository = cast(
            Callable[..., dict[str, Any]],
            _load_audit_module().audit_repository,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "module.pyc").write_bytes(b"generated")
            document = audit_repository(root, release=False)

        self.assertEqual(document["status"], "failed")
        self.assertIn(
            {
                "kind": "forbidden generated/private directory",
                "path": "__pycache__",
            },
            document["issues"],
        )


if __name__ == "__main__":
    unittest.main()
