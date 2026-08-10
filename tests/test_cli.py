from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tinygpt_forge.cli import main


class CLITests(unittest.TestCase):
    def test_smoke_prints_a_checked_forward_backward_contract(self) -> None:
        config = """
[model]
vocab_size = 17
max_seq_len = 8
d_model = 8
n_layers = 1
n_heads = 2
n_kv_heads = 1
d_ff = 16
dropout = 0.0
attention_backend = "sdpa"
""".strip()
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "smoke.toml"
            config_path.write_text(config + "\n", encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                return_code = main(["smoke", "--config", str(config_path), "--seed", "7"])

        document = json.loads(output.getvalue())
        self.assertEqual(return_code, 0)
        self.assertEqual(document["status"], "ok")
        self.assertEqual(document["seed"], 7)
        self.assertEqual(document["input_shape"], [2, 8])
        self.assertEqual(document["logits_shape"], [2, 8, 17])
        self.assertLessEqual(document["manual_sdpa_max_abs_error"], 5e-5)
        self.assertGreater(document["loss"], 0.0)

    def test_external_chat_refuses_before_reading_environment_or_network(self) -> None:
        error = io.StringIO()
        with (
            patch("tinygpt_forge.cli.OpenAICompatibleConfig.from_environment") as from_environment,
            redirect_stderr(error),
        ):
            return_code = main(["external-chat", "--prompt", "offline test"])

        self.assertEqual(return_code, 2)
        self.assertIn("cost acknowledgement", error.getvalue())
        from_environment.assert_not_called()


if __name__ == "__main__":
    unittest.main()
