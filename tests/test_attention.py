from __future__ import annotations

import copy
import unittest

import torch

from tinygpt_forge.config import ModelConfig
from tinygpt_forge.model.attention import CausalSelfAttention
from tinygpt_forge.model.rope import RotaryEmbedding


def attention_config(*, n_kv_heads: int = 2) -> ModelConfig:
    return ModelConfig(
        vocab_size=19,
        max_seq_len=16,
        d_model=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=n_kv_heads,
        d_ff=32,
        dropout=0.0,
    )


class RotaryEmbeddingTests(unittest.TestCase):
    def test_preserves_pairwise_norm(self) -> None:
        torch.manual_seed(1)
        rope = RotaryEmbedding(8).double()
        x = torch.randn(2, 3, 7, 8, dtype=torch.float64)
        rotated = rope(x)
        torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-12, rtol=1e-12)

    def test_global_position_shift_preserves_attention_dot_products(self) -> None:
        torch.manual_seed(2)
        rope = RotaryEmbedding(8).double()
        query = torch.randn(1, 2, 6, 8, dtype=torch.float64)
        key = torch.randn(1, 2, 6, 8, dtype=torch.float64)
        positions = torch.arange(6)
        shifted = positions + 137
        base_scores = rope(query, positions) @ rope(key, positions).transpose(-2, -1)
        shifted_scores = rope(query, shifted) @ rope(key, shifted).transpose(-2, -1)
        torch.testing.assert_close(base_scores, shifted_scores, atol=2e-12, rtol=2e-12)


class CausalSelfAttentionTests(unittest.TestCase):
    def test_manual_and_sdpa_forward_and_backward_match(self) -> None:
        torch.manual_seed(3)
        manual_module = CausalSelfAttention(attention_config()).double()
        sdpa_module = copy.deepcopy(manual_module)
        manual_x = torch.randn(2, 7, 16, dtype=torch.float64, requires_grad=True)
        sdpa_x = manual_x.detach().clone().requires_grad_(True)

        manual_output = manual_module(manual_x, backend="manual").hidden_states
        sdpa_output = sdpa_module(sdpa_x, backend="sdpa").hidden_states
        torch.testing.assert_close(manual_output, sdpa_output, atol=1e-10, rtol=1e-10)

        manual_output.square().sum().backward()
        sdpa_output.square().sum().backward()
        torch.testing.assert_close(manual_x.grad, sdpa_x.grad, atol=1e-9, rtol=1e-9)
        for (manual_name, manual_parameter), (sdpa_name, sdpa_parameter) in zip(
            manual_module.named_parameters(), sdpa_module.named_parameters(), strict=True
        ):
            self.assertEqual(manual_name, sdpa_name)
            torch.testing.assert_close(
                manual_parameter.grad,
                sdpa_parameter.grad,
                atol=1e-9,
                rtol=1e-9,
            )

    def test_manual_probabilities_are_causal(self) -> None:
        torch.manual_seed(4)
        module = CausalSelfAttention(attention_config()).eval()
        x = torch.randn(2, 6, 16)
        result = module(x, backend="manual", return_attention=True)
        self.assertEqual(result.hidden_states.shape, (2, 6, 16))
        probabilities = result.probabilities
        self.assertIsNotNone(probabilities)
        assert probabilities is not None
        self.assertEqual(probabilities.shape, (2, 4, 6, 6))
        upper = torch.ones(6, 6, dtype=torch.bool).triu(diagonal=1)
        self.assertEqual(torch.count_nonzero(probabilities[..., upper]).item(), 0)
        torch.testing.assert_close(
            probabilities.sum(dim=-1),
            torch.ones(2, 4, 6),
        )

    def test_future_inputs_do_not_change_past_outputs(self) -> None:
        torch.manual_seed(5)
        module = CausalSelfAttention(attention_config()).eval()
        original = torch.randn(1, 8, 16)
        changed = original.clone()
        changed[:, 5:, :] = torch.randn_like(changed[:, 5:, :])
        for backend in ("manual", "sdpa"):
            first = module(original, backend=backend).hidden_states
            second = module(changed, backend=backend).hidden_states
            torch.testing.assert_close(first[:, :5], second[:, :5], atol=1e-6, rtol=1e-6)

    def test_mha_and_gqa_return_the_same_external_shape(self) -> None:
        torch.manual_seed(6)
        x = torch.randn(2, 9, 16)
        for n_kv_heads in (1, 2, 4):
            module = CausalSelfAttention(attention_config(n_kv_heads=n_kv_heads))
            output = module(x, backend="sdpa").hidden_states
            self.assertEqual(output.shape, x.shape)

    def test_sdpa_does_not_claim_to_return_probabilities(self) -> None:
        module = CausalSelfAttention(attention_config())
        with self.assertRaisesRegex(ValueError, "manual backend"):
            module(torch.randn(1, 3, 16), backend="sdpa", return_attention=True)


if __name__ == "__main__":
    unittest.main()
