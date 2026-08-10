from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import asdict
from pathlib import Path

from tinygpt_forge.config import ModelConfig
from tinygpt_forge.training import TrainingConfig

ROOT = Path(__file__).resolve().parents[1]


class RecordedTrainingArtifactTests(unittest.TestCase):
    def test_gpu_smoke_artifact_matches_public_inputs(self) -> None:
        artifact = json.loads(
            (ROOT / "results" / "training" / "rtx5060_bf16_smoke.json").read_text(encoding="utf-8")
        )
        self.assertEqual(artifact["format"], "tinygpt-forge-gpu-smoke-v1")
        self.assertEqual(artifact["status"], "completed")

        corpus = (ROOT / "examples" / "tiny_corpus.txt").read_bytes()
        self.assertEqual(artifact["source"]["sha256"], hashlib.sha256(corpus).hexdigest())

        config_path = ROOT / "configs" / "tiny_gpu_smoke.toml"
        training_config = TrainingConfig.from_toml(config_path)
        self.assertEqual(artifact["training_config"], asdict(training_config))
        model_config = ModelConfig.from_toml(config_path).to_dict()
        resolved_model_config = artifact["model_config"].copy()
        resolved_model_config["vocab_size"] = model_config["vocab_size"]
        self.assertEqual(resolved_model_config, model_config)

        inference = artifact["inference"]
        self.assertTrue(inference["output"].startswith(inference["prompt"]))
        self.assertEqual(
            len(inference["output"]),
            len(inference["prompt"]) + inference["max_new_tokens"],
        )
        self.assertEqual(artifact["environment"]["device"], "cuda")
        self.assertTrue(artifact["environment"]["fused_optimizer_effective"])


if __name__ == "__main__":
    unittest.main()
