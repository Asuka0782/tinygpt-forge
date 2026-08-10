#!/usr/bin/env python3
"""Render the checked-in RTX 5060 p50 speedup comparison from benchmark JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "benchmarks"
INPUTS = (
    ("3.15M\nB1 · D64", "rtx5060_fp32_medium_p128_d64_static_v3.json"),
    ("24.65M\nB1 · D16", "rtx5060_fp32_large_p128_d16_static_v3.json"),
    ("24.65M\nB8 · D8", "rtx5060_fp32_large_b8_p128_d8_static_v3.json"),
)

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
GRAY = "#666666"


def _load_result(filename: str) -> dict[str, Any]:
    document = json.loads((RESULTS / filename).read_text(encoding="utf-8"))
    if document.get("format") != "tinygpt-forge-benchmark-v2":
        raise ValueError(f"unsupported benchmark format in {filename}")
    if document.get("environment", {}).get("device_name") != "NVIDIA GeForce RTX 5060 Laptop GPU":
        raise ValueError(f"unexpected benchmark device in {filename}")
    if int(document.get("method", {}).get("repeats", 0)) < 10:
        raise ValueError(f"expected at least ten retained repeats in {filename}")
    return document


def _annotate(ax: Any, bars: Any, *, inside: bool = False) -> None:
    for bar in bars:
        value = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.025 if inside else value + 0.018,
            f"{value:.2f}×",
            ha="center",
            va="top" if inside else "bottom",
            fontsize=7.5,
            color="white" if inside else "#333333",
            fontweight="bold" if inside else "normal",
        )


def main() -> None:
    records = [_load_result(filename) for _, filename in INPUTS]
    labels = [label for label, _ in INPUTS]
    x = np.arange(len(labels))

    prefill = [record["results"]["sdpa_over_manual_p50_speedup"] for record in records]
    dynamic_end = [record["results"]["dynamic_cache_over_full_p50_speedup"] for record in records]
    static_end = [record["results"]["static_cache_over_full_p50_speedup"] for record in records]
    dynamic_one = [
        record["results"]["dynamic_single_token_over_full_p50_speedup"] for record in records
    ]
    static_one = [
        record["results"]["static_single_token_over_full_p50_speedup"] for record in records
    ]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linestyle": "-",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.25), sharey=True)
    width = 0.34

    bars = axes[0].bar(x, prefill, width=0.52, color=BLUE, label="SDPA / manual")
    _annotate(axes[0], bars)
    axes[0].set_title("(a) Attention prefill")
    axes[0].set_ylabel("p50 speedup (higher is faster)")
    axes[0].legend(loc="upper left")

    bars_dynamic = axes[1].bar(
        x - width / 2, dynamic_end, width, color=ORANGE, label="Dynamic / full"
    )
    bars_static = axes[1].bar(x + width / 2, static_end, width, color=GREEN, label="Static / full")
    _annotate(axes[1], bars_dynamic)
    _annotate(axes[1], bars_static, inside=True)
    axes[1].set_title("(b) End-to-end decode")
    axes[1].legend(loc="upper left")

    bars_dynamic = axes[2].bar(
        x - width / 2, dynamic_one, width, color=ORANGE, label="Dynamic / full"
    )
    bars_static = axes[2].bar(x + width / 2, static_one, width, color=GREEN, label="Static / full")
    _annotate(axes[2], bars_dynamic)
    _annotate(axes[2], bars_static, inside=True)
    axes[2].set_title("(c) One-token decode")
    axes[2].legend(loc="upper left")

    for ax in axes:
        ax.axhline(1.0, color=GRAY, linewidth=1.0, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.76, 1.47)
        ax.set_axisbelow(True)

    fig.suptitle("RTX 5060 Laptop GPU · FP32 · p50 of 10–20 retained runs", y=1.02, fontsize=11)
    fig.text(
        0.5,
        -0.04,
        "P=128 for all cases. B=batch, D=new tokens. Random weights; systems, not quality results.",
        ha="center",
        color="#444444",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig_rtx5060_speedups.pdf")
    fig.savefig(ROOT / "figures" / "fig_rtx5060_speedups.png", dpi=300)


if __name__ == "__main__":
    main()
