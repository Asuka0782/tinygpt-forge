from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from tinygpt_forge.cache import StaticKVCache
from tinygpt_forge.config import ModelConfig
from tinygpt_forge.generation import GenerationConfig, generate
from tinygpt_forge.losses import (
    aligned_next_token_cross_entropy,
    shifted_next_token_cross_entropy,
)
from tinygpt_forge.model.gpt import TinyGPT


def model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=17,
        max_seq_len=12,
        d_model=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
        dropout=0.0,
        tie_embeddings=True,
    )


class LossTests(unittest.TestCase):
    def test_shifted_loss_matches_manual_flattening(self) -> None:
        torch.manual_seed(7)
        logits = torch.randn(2, 5, 11, dtype=torch.float64)
        token_ids = torch.randint(0, 11, (2, 5))
        expected = F.cross_entropy(logits[:, :-1].reshape(-1, 11), token_ids[:, 1:].reshape(-1))
        actual = shifted_next_token_cross_entropy(logits, token_ids)
        torch.testing.assert_close(actual, expected)

    def test_aligned_loss_rejects_mismatched_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "must match"):
            aligned_next_token_cross_entropy(
                torch.randn(2, 4, 7), torch.zeros(2, 3, dtype=torch.long)
            )


class TinyGPTTests(unittest.TestCase):
    def test_forward_loss_and_attention_shapes(self) -> None:
        torch.manual_seed(8)
        model = TinyGPT(model_config()).eval()
        input_ids = torch.randint(0, 17, (2, 10))
        targets = torch.randint(0, 17, (2, 10))
        output = model(
            input_ids,
            targets=targets,
            backend="manual",
            return_attentions=True,
        )
        self.assertEqual(output.logits.shape, (2, 10, 17))
        self.assertIsNotNone(output.loss)
        self.assertIsNotNone(output.attentions)
        assert output.attentions is not None
        self.assertEqual(len(output.attentions), 2)
        self.assertEqual(output.attentions[0].shape, (2, 4, 10, 10))

    def test_full_model_manual_and_sdpa_match(self) -> None:
        torch.manual_seed(9)
        model = TinyGPT(model_config()).eval()
        input_ids = torch.randint(0, 17, (2, 11))
        with torch.no_grad():
            manual = model(input_ids, backend="manual").logits
            sdpa = model(input_ids, backend="sdpa").logits
        torch.testing.assert_close(manual, sdpa, atol=2e-6, rtol=2e-5)

    def test_full_model_is_causal(self) -> None:
        torch.manual_seed(10)
        model = TinyGPT(model_config()).eval()
        original = torch.randint(0, 17, (1, 10))
        changed = original.clone()
        changed[:, 6:] = torch.randint(0, 17, (1, 4))
        for backend in ("manual", "sdpa"):
            with torch.no_grad():
                first = model(original, backend=backend).logits
                second = model(changed, backend=backend).logits
            torch.testing.assert_close(first[:, :6], second[:, :6], atol=2e-6, rtol=2e-5)

    def test_future_token_gradient_is_zero(self) -> None:
        torch.manual_seed(11)
        model = TinyGPT(model_config()).eval()
        input_ids = torch.randint(0, 17, (1, 8))
        embeddings = model.token_embedding(input_ids).detach().requires_grad_(True)

        hidden = embeddings
        for block in model.blocks:
            hidden, _, _ = block(hidden, backend="manual")
        logits = model.lm_head(model.final_norm(hidden))
        logits[:, 3].sum().backward()
        self.assertEqual(torch.count_nonzero(embeddings.grad[:, 4:]).item(), 0)

    def test_tied_embeddings_share_storage_and_count_once(self) -> None:
        model = TinyGPT(model_config())
        self.assertIs(model.token_embedding.weight, model.lm_head.weight)
        unique_count = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(model.parameter_count(), unique_count)

    def test_rejects_sequence_longer_than_config(self) -> None:
        model = TinyGPT(model_config())
        with self.assertRaisesRegex(ValueError, "exceeds max_seq_len"):
            model(torch.zeros(1, 13, dtype=torch.long))

    def test_incremental_cache_matches_full_sequence(self) -> None:
        torch.manual_seed(12)
        model = TinyGPT(model_config()).eval()
        input_ids = torch.randint(0, 17, (2, 9))
        with torch.no_grad():
            full = model(input_ids, backend="sdpa").logits
            prefix = model(input_ids[:, :4], backend="sdpa", use_cache=True)
            self.assertIsNotNone(prefix.past_key_values)
            pieces = [prefix.logits]
            cache = prefix.past_key_values
            for position in range(4, input_ids.size(1)):
                step = model(
                    input_ids[:, position : position + 1],
                    backend="sdpa",
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = step.past_key_values
                pieces.append(step.logits)
            cached = torch.cat(pieces, dim=1)
        torch.testing.assert_close(full, cached, atol=3e-6, rtol=2e-5)
        self.assertIsNotNone(cache)
        assert cache is not None
        for key, value in cache:
            self.assertEqual(key.shape, (2, 2, 9, 4))
            self.assertEqual(value.shape, key.shape)

    def test_greedy_generation_matches_with_and_without_cache(self) -> None:
        torch.manual_seed(13)
        model = TinyGPT(model_config()).eval()
        prompt = torch.randint(0, 17, (2, 4))
        generation = GenerationConfig(max_new_tokens=6, temperature=0.0, seed=99)
        full = generate(model, prompt, generation, backend="sdpa", use_cache=False)
        cached = generate(model, prompt, generation, backend="sdpa", use_cache=True)
        static = generate(
            model,
            prompt,
            generation,
            backend="sdpa",
            use_cache=True,
            cache_implementation="static",
        )
        torch.testing.assert_close(full, cached, atol=0, rtol=0)
        torch.testing.assert_close(full, static, atol=0, rtol=0)

    def test_static_cache_matches_full_sequence_and_uses_kv_head_storage(self) -> None:
        torch.manual_seed(15)
        model = TinyGPT(model_config()).eval()
        input_ids = torch.randint(0, 17, (2, 9))
        cache = StaticKVCache(
            model.config,
            batch_size=2,
            capacity=9,
            device="cpu",
            dtype=torch.float32,
        )
        with torch.no_grad():
            full = model(input_ids, backend="sdpa").logits
            pieces = [
                model(
                    input_ids[:, :4],
                    backend="sdpa",
                    use_cache=True,
                    static_cache=cache,
                ).logits
            ]
            for position in range(4, input_ids.size(1)):
                pieces.append(
                    model(
                        input_ids[:, position : position + 1],
                        backend="sdpa",
                        use_cache=True,
                        static_cache=cache,
                    ).logits
                )
            incremental = torch.cat(pieces, dim=1)
        torch.testing.assert_close(full, incremental, atol=3e-6, rtol=2e-5)
        self.assertEqual(cache.length, 9)
        key_storage, value_storage = cache.layer_storage(0)
        self.assertEqual(key_storage.shape, (2, 2, 9, 4))
        self.assertEqual(value_storage.shape, key_storage.shape)
        expected_bytes = (
            2 * 2 * 2 * 2 * 9 * 4 * torch.tensor([], dtype=torch.float32).element_size()
        )
        self.assertEqual(cache.allocated_bytes, expected_bytes)

    def test_cache_rejects_mismatched_device_or_dtype(self) -> None:
        model = TinyGPT(model_config()).eval()
        static = StaticKVCache(
            model.config,
            batch_size=1,
            capacity=4,
            device="cpu",
            dtype=torch.float32,
        )
        key = torch.zeros(1, 2, 1, 4, dtype=torch.float32)
        value = torch.zeros(1, 2, 1, 4, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "device, or dtype"):
            static.update(0, key, value, start_position=0)

        input_ids = torch.randint(0, model.config.vocab_size, (1, 2))
        with torch.no_grad():
            prefix = model(input_ids[:, :1], use_cache=True)
            assert prefix.past_key_values is not None
            wrong_dtype = tuple(
                (past_key.double(), past_value.double())
                for past_key, past_value in prefix.past_key_values
            )
            with self.assertRaisesRegex(ValueError, "device and dtype"):
                model(
                    input_ids[:, 1:],
                    past_key_values=wrong_dtype,
                    use_cache=True,
                )


if __name__ == "__main__":
    unittest.main()
