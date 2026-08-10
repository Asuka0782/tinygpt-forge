"""Reproducible end-to-end prefill and decode microbenchmarks."""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.profiler import ProfilerActivity, profile

from tinygpt_forge.cache import StaticKVCache
from tinygpt_forge.config import ModelConfig
from tinygpt_forge.generation import GenerationConfig, generate
from tinygpt_forge.model.gpt import TinyGPT


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _summarize(samples_ms: list[float], peak_memory_bytes: int | None) -> dict[str, Any]:
    return {
        "unit": "ms",
        "repeats": len(samples_ms),
        "mean": statistics.fmean(samples_ms),
        "stdev": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "p50": statistics.median(samples_ms),
        "p90": _percentile(samples_ms, 0.9),
        "min": min(samples_ms),
        "max": max(samples_ms),
        "peak_memory_bytes": peak_memory_bytes,
        "raw_samples_ms": samples_ms,
    }


def _measure(
    function: Callable[[], Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        function()
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    samples_ms: list[float] = []
    for _ in range(repeats):
        _synchronize(device)
        start = time.perf_counter_ns()
        result = function()
        _synchronize(device)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        samples_ms.append(elapsed_ms)
        if result.numel() == 0:
            raise RuntimeError("benchmark function returned an empty tensor")
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return _summarize(samples_ms, peak_memory)


def _sdpa_operator_names(model: TinyGPT, input_ids: Tensor) -> list[str]:
    with profile(activities=[ProfilerActivity.CPU]) as trace:
        with torch.inference_mode():
            model(input_ids, backend="sdpa")
        _synchronize(input_ids.device)
    markers = ("scaled_dot_product", "flash_attention", "efficient_attention")
    return sorted(
        {
            event.key
            for event in trace.key_averages()
            if any(marker in event.key for marker in markers)
        }
    )


def run_benchmark(
    *,
    model_config: ModelConfig,
    device: torch.device | str,
    dtype_name: str,
    batch_size: int,
    prompt_length: int,
    new_tokens: int,
    warmup: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Compare manual/SDPA prefill and full/dynamic-cache greedy decode."""

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if dtype_name not in {"float32", "bfloat16", "float16"}:
        raise ValueError("dtype must be float32, bfloat16, or float16")
    if target.type == "cpu" and dtype_name == "float16":
        raise ValueError("float16 is not supported by the CPU benchmark")
    if batch_size <= 0 or prompt_length <= 0 or new_tokens <= 0:
        raise ValueError("batch_size, prompt_length, and new_tokens must be positive")
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats positive")
    if prompt_length + new_tokens > model_config.max_seq_len:
        raise ValueError("prompt_length + new_tokens exceeds max_seq_len")

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_name]
    torch.manual_seed(seed)
    if target.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = TinyGPT(model_config).to(device=target, dtype=dtype).eval()
    device_properties = torch.cuda.get_device_properties(target) if target.type == "cuda" else None
    input_ids = torch.randint(
        0,
        model_config.vocab_size,
        (batch_size, prompt_length),
        device=target,
    )
    generation_config = GenerationConfig(max_new_tokens=new_tokens, temperature=0.0, seed=seed)

    with torch.inference_mode():
        manual_logits = model(input_ids, backend="manual").logits
        sdpa_logits = model(input_ids, backend="sdpa").logits
        dynamic_prefix_output = model(input_ids, backend="sdpa", use_cache=True)
        if dynamic_prefix_output.past_key_values is None:
            raise RuntimeError("dynamic prefix did not return a cache")
        dynamic_prefix = dynamic_prefix_output.past_key_values
        next_token = sdpa_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        full_step_input = torch.cat((input_ids, next_token), dim=1)
        full_step_logits = model(full_step_input, backend="sdpa").logits[:, -1, :]
        dynamic_step_logits = model(
            next_token,
            backend="sdpa",
            past_key_values=dynamic_prefix,
            use_cache=True,
        ).logits[:, -1, :]
        parameter = next(model.parameters())
        static_prefix = StaticKVCache(
            model.config,
            batch_size=batch_size,
            capacity=prompt_length + 1,
            device=target,
            dtype=parameter.dtype,
        )
        model(input_ids, backend="sdpa", use_cache=True, static_cache=static_prefix)
        static_step_logits = model(
            next_token,
            backend="sdpa",
            use_cache=True,
            static_cache=static_prefix,
        ).logits[:, -1, :]
        full_ids = generate(
            model,
            input_ids,
            generation_config,
            backend="sdpa",
            use_cache=False,
        )
        cached_ids = generate(
            model,
            input_ids,
            generation_config,
            backend="sdpa",
            use_cache=True,
            cache_implementation="dynamic",
        )
        static_ids = generate(
            model,
            input_ids,
            generation_config,
            backend="sdpa",
            use_cache=True,
            cache_implementation="static",
        )
    _synchronize(target)
    forward_error = (manual_logits.float() - sdpa_logits.float()).abs().max().item()
    dynamic_step_error = (full_step_logits.float() - dynamic_step_logits.float()).abs().max().item()
    static_step_error = (full_step_logits.float() - static_step_logits.float()).abs().max().item()
    correctness_tolerance = {
        "float32": 5e-5,
        "bfloat16": 5e-2,
        "float16": 5e-2,
    }[dtype_name]
    largest_error = max(forward_error, dynamic_step_error, static_step_error)
    if not math.isfinite(largest_error) or largest_error > correctness_tolerance:
        raise RuntimeError(
            "manual/SDPA/cache correctness error "
            f"{largest_error:.6g} exceeds tolerance {correctness_tolerance:.6g}"
        )
    greedy_exact = torch.equal(full_ids, cached_ids)
    static_greedy_exact = torch.equal(full_ids, static_ids)
    if not greedy_exact or not static_greedy_exact:
        raise RuntimeError("full, dynamic, and static greedy generation diverged before timing")

    static_prefix.rewind(prompt_length)

    def static_single_token_step() -> Tensor:
        static_prefix.rewind(prompt_length)
        return model.forward(
            next_token,
            backend="sdpa",
            use_cache=True,
            static_cache=static_prefix,
        ).logits

    with torch.inference_mode():
        manual_prefill = _measure(
            lambda: model(input_ids, backend="manual").logits,
            device=target,
            warmup=warmup,
            repeats=repeats,
        )
        sdpa_prefill = _measure(
            lambda: model(input_ids, backend="sdpa").logits,
            device=target,
            warmup=warmup,
            repeats=repeats,
        )
        full_decode = _measure(
            lambda: generate(
                model,
                input_ids,
                generation_config,
                backend="sdpa",
                use_cache=False,
            ),
            device=target,
            warmup=warmup,
            repeats=repeats,
        )
        cached_decode = _measure(
            lambda: generate(
                model,
                input_ids,
                generation_config,
                backend="sdpa",
                use_cache=True,
                cache_implementation="dynamic",
            ),
            device=target,
            warmup=warmup,
            repeats=repeats,
        )
        static_decode = _measure(
            lambda: generate(
                model,
                input_ids,
                generation_config,
                backend="sdpa",
                use_cache=True,
                cache_implementation="static",
            ),
            device=target,
            warmup=warmup,
            repeats=repeats,
        )
        full_single_token = _measure(
            lambda: model(full_step_input, backend="sdpa").logits,
            device=target,
            warmup=warmup,
            repeats=repeats,
        )
        dynamic_single_token = _measure(
            lambda: (
                model(
                    next_token,
                    backend="sdpa",
                    past_key_values=dynamic_prefix,
                    use_cache=True,
                ).logits
            ),
            device=target,
            warmup=warmup,
            repeats=repeats,
        )
        static_single_token = _measure(
            static_single_token_step,
            device=target,
            warmup=warmup,
            repeats=repeats,
        )

    sdpa_p50 = sdpa_prefill["p50"]
    cached_p50 = cached_decode["p50"]
    static_p50 = static_decode["p50"]
    document: dict[str, Any] = {
        "format": "tinygpt-forge-benchmark-v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "model_initialization": "random",
        "parameter_count": model.parameter_count(),
        "model_config": model_config.to_dict(),
        "shape": {
            "batch_size": batch_size,
            "prompt_length": prompt_length,
            "new_tokens": new_tokens,
        },
        "method": {
            "warmup": warmup,
            "repeats": repeats,
            "synchronization": "before and after every timed CUDA sample",
            "decode": "end-to-end greedy generation including Python and token concatenation",
            "single_token_decode": (
                "one model call after a prefilled prompt; excludes prefill and sampling"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": str(target),
            "device_name": torch.cuda.get_device_name(target) if target.type == "cuda" else "CPU",
            "dtype": dtype_name,
            "cuda_runtime": torch.version.cuda if target.type == "cuda" else None,
            "cudnn_version": (  # PyTorch leaves this public function untyped.
                torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
                if target.type == "cuda"
                else None
            ),
            "compute_capability": (
                f"{device_properties.major}.{device_properties.minor}"
                if device_properties is not None
                else None
            ),
            "device_total_memory_bytes": (
                device_properties.total_memory if device_properties is not None else None
            ),
            "multiprocessor_count": (
                device_properties.multi_processor_count if device_properties is not None else None
            ),
            "flash_attention_compiled": (
                torch.backends.cuda.is_flash_attention_available()
                if target.type == "cuda"
                else False
            ),
            "sdpa_operator_names": (
                _sdpa_operator_names(model, input_ids)
                if target.type == "cuda"
                else ["not-profiled-on-cpu"]
            ),
        },
        "correctness": {
            "manual_sdpa_max_abs_error": forward_error,
            "max_abs_tolerance": correctness_tolerance,
            "full_cache_greedy_ids_exact": greedy_exact,
            "full_static_cache_greedy_ids_exact": static_greedy_exact,
            "dynamic_single_token_max_abs_error": dynamic_step_error,
            "static_single_token_max_abs_error": static_step_error,
        },
        "results": {
            "manual_prefill": manual_prefill,
            "sdpa_prefill": sdpa_prefill,
            "full_decode": full_decode,
            "dynamic_cache_decode": cached_decode,
            "static_cache_decode": static_decode,
            "full_single_token_decode": full_single_token,
            "dynamic_single_token_decode": dynamic_single_token,
            "static_single_token_decode": static_single_token,
            "sdpa_over_manual_p50_speedup": manual_prefill["p50"] / sdpa_p50,
            "dynamic_cache_over_full_p50_speedup": full_decode["p50"] / cached_p50,
            "static_cache_over_full_p50_speedup": full_decode["p50"] / static_p50,
            "dynamic_single_token_over_full_p50_speedup": (
                full_single_token["p50"] / dynamic_single_token["p50"]
            ),
            "static_single_token_over_full_p50_speedup": (
                full_single_token["p50"] / static_single_token["p50"]
            ),
            "full_decode_tokens_per_second_p50": batch_size
            * new_tokens
            / (full_decode["p50"] / 1000),
            "dynamic_cache_tokens_per_second_p50": (
                batch_size * new_tokens / (cached_decode["p50"] / 1000)
            ),
            "static_cache_tokens_per_second_p50": (
                batch_size * new_tokens / (static_decode["p50"] / 1000)
            ),
        },
        "limitations": [
            "This is a randomly initialized small model; it measures mechanisms, not quality.",
            "One machine and one shape do not establish a general performance claim.",
            "Dynamic cache concatenates tensors and can be slower for tiny workloads.",
            "Static cache allocation is included in each end-to-end generation sample.",
            "Power, temperature, and background-process variance are not controlled here.",
        ],
    }
    return document


def save_benchmark(document: Mapping[str, Any], path: str | Path) -> None:
    """Atomically save the full benchmark document, including raw samples."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
