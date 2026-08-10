from __future__ import annotations

import unittest

from tinygpt_forge.benchmark import run_benchmark
from tinygpt_forge.config import ModelConfig


class BenchmarkTests(unittest.TestCase):
    def test_cpu_benchmark_keeps_raw_samples_and_correctness_gate(self) -> None:
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
        document = run_benchmark(
            model_config=config,
            device="cpu",
            dtype_name="float32",
            batch_size=1,
            prompt_length=3,
            new_tokens=2,
            warmup=0,
            repeats=2,
            seed=42,
        )
        self.assertTrue(document["correctness"]["full_cache_greedy_ids_exact"])
        self.assertTrue(document["correctness"]["full_static_cache_greedy_ids_exact"])
        self.assertEqual(document["correctness"]["max_abs_tolerance"], 5e-5)
        self.assertLess(
            document["correctness"]["manual_sdpa_max_abs_error"],
            document["correctness"]["max_abs_tolerance"],
        )
        self.assertLess(document["correctness"]["dynamic_single_token_max_abs_error"], 1e-5)
        self.assertLess(document["correctness"]["static_single_token_max_abs_error"], 1e-5)
        self.assertEqual(len(document["results"]["manual_prefill"]["raw_samples_ms"]), 2)
        self.assertEqual(len(document["results"]["dynamic_cache_decode"]["raw_samples_ms"]), 2)
        self.assertEqual(len(document["results"]["static_cache_decode"]["raw_samples_ms"]), 2)
        self.assertEqual(
            len(document["results"]["static_single_token_decode"]["raw_samples_ms"]), 2
        )
        self.assertIsNone(document["results"]["sdpa_prefill"]["peak_memory_bytes"])
        self.assertIsNone(document["environment"]["cuda_runtime"])
        self.assertIsNone(document["environment"]["compute_capability"])
        self.assertIsNone(document["environment"]["device_total_memory_bytes"])


if __name__ == "__main__":
    unittest.main()
