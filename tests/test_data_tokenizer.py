from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from tinygpt_forge.data import NextTokenBatcher, split_token_ids
from tinygpt_forge.tokenizer import CharacterTokenizer


class CharacterTokenizerTests(unittest.TestCase):
    def test_roundtrip_and_deterministic_vocabulary(self) -> None:
        text = "TinyGPT：你好\nabcabc"
        first = CharacterTokenizer.train(text)
        second = CharacterTokenizer.train(text[::-1])
        self.assertEqual(first.tokens, second.tokens)
        self.assertEqual(first.decode(first.encode(text, strict=True)), text)
        self.assertEqual(first.fingerprint(), second.fingerprint())

    def test_unknown_character_has_explicit_behavior(self) -> None:
        tokenizer = CharacterTokenizer.train("abc")
        self.assertEqual(tokenizer.encode("a?"), [tokenizer.tokens.index("a"), tokenizer.unk_id])
        with self.assertRaisesRegex(ValueError, "out-of-vocabulary"):
            tokenizer.encode("?", strict=True)

    def test_json_save_and_load(self) -> None:
        tokenizer = CharacterTokenizer.train("保存与重载")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tokenizer.json"
            tokenizer.save(path)
            loaded = CharacterTokenizer.load(path)
        self.assertEqual(tokenizer, loaded)


class DataTests(unittest.TestCase):
    def test_contiguous_train_validation_test_split(self) -> None:
        splits = split_token_ids(list(range(100)), validation_fraction=0.2, test_fraction=0.1)
        self.assertEqual(splits.train.tolist(), list(range(70)))
        self.assertEqual(splits.validation.tolist(), list(range(70, 90)))
        self.assertEqual(splits.test.tolist(), list(range(90, 100)))

    def test_batch_inputs_and_targets_are_shifted_once(self) -> None:
        tokens = torch.arange(100, dtype=torch.long)
        batcher = NextTokenBatcher(tokens, block_size=8, batch_size=4, seed=42)
        inputs, targets = batcher.next_batch()
        torch.testing.assert_close(inputs[:, 1:], targets[:, :-1])

    def test_batcher_rng_state_supports_exact_resume(self) -> None:
        tokens = torch.arange(100, dtype=torch.long)
        first = NextTokenBatcher(tokens, block_size=8, batch_size=4, seed=42)
        first.next_batch()
        state = first.get_rng_state()
        expected = first.next_batch()

        resumed = NextTokenBatcher(tokens, block_size=8, batch_size=4, seed=999)
        resumed.set_rng_state(state)
        actual = resumed.next_batch()
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])


if __name__ == "__main__":
    unittest.main()
