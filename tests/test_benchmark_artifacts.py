from __future__ import annotations

import json
import statistics
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "benchmarks"
ARTIFACTS = (
    "rtx5060_fp32_medium_p128_d64_static_v3.json",
    "rtx5060_fp32_large_p128_d16_static_v3.json",
    "rtx5060_fp32_large_b8_p128_d8_static_v3.json",
)


class RecordedBenchmarkArtifactTests(unittest.TestCase):
    def _load(self, name: str) -> dict[str, Any]:
        return json.loads((RESULTS / name).read_text(encoding="utf-8"))

    def test_recorded_samples_and_summaries_are_self_consistent(self) -> None:
        timed_metrics = (
            "manual_prefill",
            "sdpa_prefill",
            "full_decode",
            "dynamic_cache_decode",
            "static_cache_decode",
            "full_single_token_decode",
            "dynamic_single_token_decode",
            "static_single_token_decode",
        )
        for name in ARTIFACTS:
            with self.subTest(name=name):
                document = self._load(name)
                self.assertEqual(document["format"], "tinygpt-forge-benchmark-v2")
                self.assertEqual(
                    document["environment"]["device_name"],
                    "NVIDIA GeForce RTX 5060 Laptop GPU",
                )
                repeats = document["method"]["repeats"]
                results = document["results"]
                for metric in timed_metrics:
                    samples = results[metric]["raw_samples_ms"]
                    self.assertEqual(len(samples), repeats)
                    self.assertAlmostEqual(results[metric]["p50"], statistics.median(samples))

                expected_speedups = {
                    "sdpa_over_manual_p50_speedup": (
                        "manual_prefill",
                        "sdpa_prefill",
                    ),
                    "dynamic_cache_over_full_p50_speedup": (
                        "full_decode",
                        "dynamic_cache_decode",
                    ),
                    "static_cache_over_full_p50_speedup": (
                        "full_decode",
                        "static_cache_decode",
                    ),
                    "dynamic_single_token_over_full_p50_speedup": (
                        "full_single_token_decode",
                        "dynamic_single_token_decode",
                    ),
                    "static_single_token_over_full_p50_speedup": (
                        "full_single_token_decode",
                        "static_single_token_decode",
                    ),
                }
                for summary, (baseline, optimized) in expected_speedups.items():
                    expected = results[baseline]["p50"] / results[optimized]["p50"]
                    self.assertAlmostEqual(results[summary], expected)

    def test_recorded_correctness_gates_pass(self) -> None:
        for name in ARTIFACTS:
            with self.subTest(name=name):
                correctness = self._load(name)["correctness"]
                self.assertTrue(correctness["full_cache_greedy_ids_exact"])
                self.assertTrue(correctness["full_static_cache_greedy_ids_exact"])
                self.assertLessEqual(correctness["manual_sdpa_max_abs_error"], 3e-6)
                self.assertLessEqual(correctness["dynamic_single_token_max_abs_error"], 3e-6)
                self.assertLessEqual(correctness["static_single_token_max_abs_error"], 3e-6)

    def test_rendered_figure_artifacts_exist(self) -> None:
        for name in ("fig_rtx5060_speedups.png", "fig_rtx5060_speedups.pdf"):
            with self.subTest(name=name):
                path = ROOT / "figures" / name
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
