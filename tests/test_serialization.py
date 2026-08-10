from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tinygpt_forge.serialization import MAX_METADATA_BYTES, read_json_object
from tinygpt_forge.toml_compat import load_toml


class SerializationTests(unittest.TestCase):
    def test_json_requires_unique_finite_object_metadata(self) -> None:
        invalid_documents = {
            "duplicate.json": b'{"format":"a","format":"b"}',
            "nonfinite.json": b'{"loss":NaN}',
            "array.json": b"[]",
            "invalid-utf8.json": b'{"value":"\xff"}',
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for filename, payload in invalid_documents.items():
                with self.subTest(filename=filename):
                    path = root / filename
                    path.write_bytes(payload)
                    with self.assertRaisesRegex(ValueError, "strict UTF-8 JSON|JSON object"):
                        read_json_object(path, kind="test metadata")

    def test_json_and_toml_reject_oversized_metadata_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oversized = b" " * (MAX_METADATA_BYTES + 1)
            json_path = root / "oversized.json"
            toml_path = root / "oversized.toml"
            json_path.write_bytes(oversized)
            toml_path.write_bytes(oversized)

            with self.assertRaisesRegex(ValueError, "metadata limit"):
                read_json_object(json_path, kind="test metadata")
            with self.assertRaisesRegex(ValueError, "metadata limit"):
                load_toml(toml_path)


if __name__ == "__main__":
    unittest.main()
