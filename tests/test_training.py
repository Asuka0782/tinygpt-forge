from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from tinygpt_forge.checkpoint import load_weights
from tinygpt_forge.config import ModelConfig
from tinygpt_forge.tokenizer import CharacterTokenizer
from tinygpt_forge.training import TrainingConfig, train_character_model


class TrainingTests(unittest.TestCase):
    def test_tiny_cpu_training_writes_reproducible_artifacts(self) -> None:
        corpus = ("abcd efgh\n" * 120) + ("validation symbols xyz\n" * 30)
        model_config = ModelConfig(
            vocab_size=2,
            max_seq_len=8,
            d_model=8,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            d_ff=16,
            dropout=0.0,
        )
        training_config = TrainingConfig(
            batch_size=2,
            block_size=8,
            max_steps=2,
            gradient_accumulation_steps=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            eval_interval=1,
            eval_batches=2,
            validation_fraction=0.1,
            test_fraction=0.1,
            seed=17,
            device="cpu",
            dtype="float32",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = root / "corpus.txt"
            corpus_path.write_text(corpus, encoding="utf-8")
            run_directory = root / "run"
            result = train_character_model(
                corpus_path=corpus_path,
                run_directory=run_directory,
                model_config=model_config,
                training_config=training_config,
            )
            tokenizer = CharacterTokenizer.load(run_directory / "tokenizer.json")
            restored = load_weights(run_directory / "checkpoints" / "last").eval()
            prompt = torch.tensor([tokenizer.encode("abcd")], dtype=torch.long)
            with torch.no_grad():
                logits = restored(prompt).logits

            self.assertTrue((run_directory / "run.json").is_file())
            self.assertTrue(
                (run_directory / "checkpoints" / "best" / "model.safetensors").is_file()
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(result["history"]), 3)
            self.assertEqual(logits.shape, (1, 4, tokenizer.vocab_size))
            self.assertTrue(torch.isfinite(logits).all())

            training_manifest_path = run_directory / "checkpoints" / "resume" / "training.json"
            training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
            training_manifest["trainer_state"]["step"] = True
            training_manifest_path.write_text(json.dumps(training_manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid field types"):
                train_character_model(
                    corpus_path=corpus_path,
                    run_directory=run_directory,
                    model_config=model_config,
                    training_config=replace(training_config, max_steps=3),
                    resume_from=run_directory / "checkpoints" / "resume",
                )

    def test_interrupted_resume_matches_uninterrupted_training_bitwise(self) -> None:
        corpus = "resume exactness with dropout and adam state\n" * 160
        model_config = ModelConfig(
            vocab_size=2,
            max_seq_len=8,
            d_model=8,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            d_ff=16,
            dropout=0.1,
        )

        def training_config(max_steps: int) -> TrainingConfig:
            return TrainingConfig(
                batch_size=2,
                block_size=8,
                max_steps=max_steps,
                gradient_accumulation_steps=2,
                learning_rate=1e-3,
                weight_decay=0.01,
                eval_interval=2,
                eval_batches=2,
                validation_fraction=0.1,
                test_fraction=0.1,
                seed=23,
                device="cpu",
                dtype="float32",
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = root / "corpus.txt"
            corpus_path.write_text(corpus, encoding="utf-8")

            uninterrupted_dir = root / "uninterrupted"
            uninterrupted_result = train_character_model(
                corpus_path=corpus_path,
                run_directory=uninterrupted_dir,
                model_config=model_config,
                training_config=training_config(4),
            )

            resumed_dir = root / "resumed"
            train_character_model(
                corpus_path=corpus_path,
                run_directory=resumed_dir,
                model_config=model_config,
                training_config=training_config(2),
            )
            resumed_result = train_character_model(
                corpus_path=corpus_path,
                run_directory=resumed_dir,
                model_config=model_config,
                training_config=training_config(4),
                resume_from=resumed_dir / "checkpoints" / "resume",
            )

            uninterrupted_model = load_weights(uninterrupted_dir / "checkpoints" / "last")
            resumed_model = load_weights(resumed_dir / "checkpoints" / "last")

        for expected, actual in zip(
            uninterrupted_model.parameters(), resumed_model.parameters(), strict=True
        ):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(uninterrupted_result["history"], resumed_result["history"])
        self.assertEqual(uninterrupted_result["test_loss_last"], resumed_result["test_loss_last"])
        self.assertEqual(resumed_result["resumed_from_step"], 2)


if __name__ == "__main__":
    unittest.main()
