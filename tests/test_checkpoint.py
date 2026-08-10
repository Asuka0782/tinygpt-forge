from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from tinygpt_forge.checkpoint import (
    load_training_checkpoint,
    load_weights,
    save_training_checkpoint,
    save_weights,
)
from tinygpt_forge.config import ModelConfig
from tinygpt_forge.model.gpt import TinyGPT


class CheckpointTests(unittest.TestCase):
    def test_safetensors_roundtrip_is_exact(self) -> None:
        torch.manual_seed(14)
        config = ModelConfig(
            vocab_size=23,
            max_seq_len=12,
            d_model=16,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            dropout=0.0,
            tie_embeddings=True,
        )
        model = TinyGPT(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 10))
        with torch.no_grad():
            expected = model(input_ids).logits

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "checkpoint"
            manifest = save_weights(model, directory)
            restored = load_weights(directory).eval()
            with torch.no_grad():
                actual = restored(input_ids).logits
            disk_manifest = json.loads((directory / "model.json").read_text(encoding="utf-8"))

        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(manifest, disk_manifest)
        self.assertIs(restored.token_embedding.weight, restored.lm_head.weight)

    def test_manifest_hash_detects_tampering(self) -> None:
        config = ModelConfig(
            vocab_size=11,
            max_seq_len=8,
            d_model=8,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            d_ff=16,
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "checkpoint"
            save_weights(TinyGPT(config), directory)
            manifest_path = directory / "model.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["weights_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_weights(directory)

    def test_pickle_free_training_state_roundtrip(self) -> None:
        torch.manual_seed(123)
        config = ModelConfig(
            vocab_size=13,
            max_seq_len=8,
            d_model=8,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            d_ff=16,
            dropout=0.0,
        )
        model = TinyGPT(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        inputs = torch.randint(0, config.vocab_size, (2, 6))
        targets = torch.randint(0, config.vocab_size, (2, 6))
        output = model(inputs, targets=targets)
        assert output.loss is not None
        output.loss.backward()
        optimizer.step()
        batch_generator = torch.Generator().manual_seed(99)
        batch_generator_state = batch_generator.get_state()

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "training"
            save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                batcher_rng_state=batch_generator_state,
                scaler=scaler,
                trainer_state={"step": 1, "history": [{"loss": output.loss.item()}]},
                directory=directory,
            )
            expected_random = torch.rand(5)

            restored_model = TinyGPT(config)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=9e-2)
            restored_scaler = torch.amp.GradScaler("cpu", enabled=False)
            trainer_state, restored_batcher_state = load_training_checkpoint(
                model=restored_model,
                optimizer=restored_optimizer,
                scaler=restored_scaler,
                directory=directory,
                device="cpu",
            )
            actual_random = torch.rand(5)

        for expected, actual in zip(model.parameters(), restored_model.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        expected_optimizer = optimizer.state_dict()
        actual_optimizer = restored_optimizer.state_dict()
        self.assertEqual(expected_optimizer["param_groups"], actual_optimizer["param_groups"])
        self.assertEqual(expected_optimizer["state"].keys(), actual_optimizer["state"].keys())
        for parameter_id in expected_optimizer["state"]:
            for name, expected in expected_optimizer["state"][parameter_id].items():
                actual = actual_optimizer["state"][parameter_id][name]
                if isinstance(expected, torch.Tensor):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                else:
                    self.assertEqual(actual, expected)
        self.assertEqual(trainer_state["step"], 1)
        torch.testing.assert_close(restored_batcher_state, batch_generator_state)
        torch.testing.assert_close(actual_random, expected_random, atol=0, rtol=0)

    def test_training_bundle_rejects_cross_manifest_mismatch(self) -> None:
        config = ModelConfig(
            vocab_size=11,
            max_seq_len=8,
            d_model=8,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            d_ff=16,
        )
        model = TinyGPT(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        inputs = torch.randint(0, config.vocab_size, (1, 4))
        output = model(inputs, targets=inputs)
        assert output.loss is not None
        output.loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "training"
            save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                batcher_rng_state=torch.Generator().manual_seed(7).get_state(),
                scaler=scaler,
                trainer_state={"step": 1},
                directory=directory,
            )
            training_path = directory / "training.json"
            training_manifest = json.loads(training_path.read_text(encoding="utf-8"))
            training_manifest["optimizer_tensors_sha256"] = "0" * 64
            training_path.write_text(json.dumps(training_manifest), encoding="utf-8")

            restored_model = TinyGPT(config)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
            parameters_before = [
                parameter.detach().clone() for parameter in restored_model.parameters()
            ]
            with self.assertRaisesRegex(ValueError, "training/optimizer"):
                load_training_checkpoint(
                    model=restored_model,
                    optimizer=restored_optimizer,
                    scaler=torch.amp.GradScaler("cpu", enabled=False),
                    directory=directory,
                    device="cpu",
                )
            for before, after in zip(
                parameters_before,
                restored_model.parameters(),
                strict=True,
            ):
                torch.testing.assert_close(after, before, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
