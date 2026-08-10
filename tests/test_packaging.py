from __future__ import annotations

import ast
import hashlib
import unittest
from pathlib import Path

from tinygpt_forge import __version__
from tinygpt_forge.toml_compat import load_toml

ROOT = Path(__file__).resolve().parents[1]


class PackagingContractTests(unittest.TestCase):
    def test_declared_runtime_contract_matches_release_metadata(self) -> None:
        document = load_toml(ROOT / "pyproject.toml")
        project = document["project"]

        self.assertEqual(project["version"], __version__)
        self.assertEqual(project["license"], "Apache-2.0")
        self.assertEqual(project["license-files"], ["LICENSE"])
        self.assertEqual(project["requires-python"], ">=3.10,<3.15")
        self.assertIn("torch>=2.3,<3.0", project["dependencies"])
        self.assertIn(
            "numpy>=1.26,<2.0; python_version < '3.13'",
            project["dependencies"],
        )
        self.assertIn(
            "numpy>=2.1,<3.0; python_version >= '3.13'",
            project["dependencies"],
        )
        self.assertIn("setuptools>=77.0.3", document["build-system"]["requires"])

    def test_license_is_the_canonical_apache_20_text(self) -> None:
        normalized = (ROOT / "LICENSE").read_text(encoding="utf-8").replace("\r\n", "\n")
        digest = hashlib.sha256(normalized.rstrip("\n").encode()).hexdigest()

        self.assertEqual(
            digest,
            "58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd",
        )

    def test_python_sources_parse_with_python_310_grammar(self) -> None:
        roots = (ROOT / "src", ROOT / "scripts", ROOT / "tests")
        failures: list[str] = []

        for source_root in roots:
            for path in sorted(source_root.rglob("*.py")):
                try:
                    ast.parse(
                        path.read_text(encoding="utf-8"),
                        filename=str(path),
                        feature_version=(3, 10),
                    )
                except SyntaxError as error:
                    failures.append(f"{path.relative_to(ROOT)}: {error}")

        self.assertEqual(failures, [], "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
