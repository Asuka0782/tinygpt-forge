from __future__ import annotations

import unittest

from tinygpt_forge.config import ModelConfig


def valid_config(**overrides: object) -> ModelConfig:
    values: dict[str, object] = {
        "vocab_size": 17,
        "max_seq_len": 16,
        "d_model": 16,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "d_ff": 32,
        "dropout": 0.0,
    }
    values.update(overrides)
    return ModelConfig.from_mapping(values)


class ModelConfigTests(unittest.TestCase):
    def test_derived_head_values(self) -> None:
        config = valid_config()
        self.assertEqual(config.head_dim, 4)
        self.assertEqual(config.query_groups, 2)

    def test_rejects_invalid_grouping(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_heads must be divisible"):
            valid_config(n_heads=4, n_kv_heads=3)

    def test_rejects_odd_rope_head_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "head_dim must be even"):
            valid_config(d_model=12, n_heads=4, n_kv_heads=2)

    def test_rejects_unknown_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown model config fields"):
            valid_config(typo_field=123)


if __name__ == "__main__":
    unittest.main()
