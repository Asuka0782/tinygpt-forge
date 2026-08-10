"""Minimal CLI used to validate the first model contract."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import torch

from tinygpt_forge.benchmark import run_benchmark, save_benchmark
from tinygpt_forge.checkpoint import load_weights
from tinygpt_forge.config import ModelConfig
from tinygpt_forge.generation import GenerationConfig, generate
from tinygpt_forge.model.gpt import TinyGPT
from tinygpt_forge.providers import (
    ChatMessage,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
)
from tinygpt_forge.tokenizer import CharacterTokenizer
from tinygpt_forge.training import TrainingConfig, train_character_model


def _smoke(config_path: Path, seed: int) -> int:
    torch.manual_seed(seed)
    config = ModelConfig.from_toml(config_path)
    model = TinyGPT(config)
    time = min(16, config.max_seq_len)
    input_ids = torch.randint(0, config.vocab_size, (2, time))
    targets = torch.randint(0, config.vocab_size, (2, time))

    model.eval()
    with torch.no_grad():
        manual = model(input_ids, backend="manual").logits
        sdpa = model(input_ids, backend="sdpa").logits
    max_abs_error = (manual - sdpa).abs().max().item()
    if not math.isfinite(max_abs_error) or max_abs_error > 5e-5:
        raise RuntimeError(f"manual/SDPA smoke error {max_abs_error:.6g} exceeds tolerance 5e-05")

    model.train()
    training_output = model(input_ids, targets=targets, backend=config.attention_backend)
    if training_output.loss is None:
        raise RuntimeError("smoke loss was not computed")
    training_output.loss.backward()

    result = {
        "status": "ok",
        "seed": seed,
        "torch_version": torch.__version__,
        "config": config.to_dict(),
        "parameter_count": model.parameter_count(),
        "input_shape": list(input_ids.shape),
        "logits_shape": list(sdpa.shape),
        "manual_sdpa_max_abs_error": max_abs_error,
        "loss": training_output.loss.item(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without performing side effects."""

    parser = argparse.ArgumentParser(prog="tinygpt")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="run the tiny forward/backward contract")
    smoke.add_argument("--config", type=Path, required=True)
    smoke.add_argument("--seed", type=int, default=42)

    train = subparsers.add_parser("train-char", help="train a local character model")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--text", type=Path, required=True)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--resume-from", type=Path)
    train.add_argument("--max-steps", type=int, help="override only the target max_steps")

    generate_parser = subparsers.add_parser(
        "generate-char", help="generate from a local checkpoint"
    )
    generate_parser.add_argument("--checkpoint", type=Path, required=True)
    generate_parser.add_argument("--tokenizer", type=Path, required=True)
    generate_parser.add_argument("--prompt", type=str, required=True)
    generate_parser.add_argument("--max-new-tokens", type=int, default=32)
    generate_parser.add_argument("--temperature", type=float, default=0.0)
    generate_parser.add_argument("--top-k", type=int)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")

    benchmark = subparsers.add_parser("benchmark", help="benchmark prefill and cached decode")
    benchmark.add_argument("--config", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    benchmark.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
    )
    benchmark.add_argument("--batch-size", type=int, default=1)
    benchmark.add_argument("--prompt-length", type=int, default=16)
    benchmark.add_argument("--new-tokens", type=int, default=16)
    benchmark.add_argument("--warmup", type=int, default=5)
    benchmark.add_argument("--repeats", type=int, default=20)
    benchmark.add_argument("--seed", type=int, default=42)

    external = subparsers.add_parser(
        "external-chat",
        help="call an explicitly configured OpenAI-compatible endpoint",
    )
    external.add_argument("--prompt", required=True)
    external.add_argument("--max-tokens", type=int, default=256)
    external.add_argument("--temperature", type=float, default=0.0)
    external.add_argument(
        "--yes-i-understand-this-may-cost-money",
        action="store_true",
        help="required acknowledgement before any external request",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected TinyGPT Forge command."""

    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        return _smoke(args.config, args.seed)
    if args.command == "train-char":
        training_config = TrainingConfig.from_toml(args.config)
        if args.max_steps is not None:
            training_config = replace(training_config, max_steps=args.max_steps)
        result = train_character_model(
            corpus_path=args.text,
            run_directory=args.run_dir,
            model_config=ModelConfig.from_toml(args.config),
            training_config=training_config,
            resume_from=args.resume_from,
        )
        summary = {
            "status": result["status"],
            "best_step": result["best_step"],
            "best_validation_loss": result["best_validation_loss"],
            "test_loss_last": result["test_loss_last"],
            "run_file": str(args.run_dir / "run.json"),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "generate-char":
        device = torch.device(args.device)
        model = load_weights(args.checkpoint, device=device).eval()
        tokenizer = CharacterTokenizer.load(args.tokenizer)
        if tokenizer.vocab_size != model.config.vocab_size:
            raise ValueError("tokenizer vocabulary does not match the checkpoint")
        prompt_ids = torch.tensor(
            [tokenizer.encode(args.prompt)],
            dtype=torch.long,
            device=device,
        )
        output_ids = generate(
            model,
            prompt_ids,
            GenerationConfig(
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=args.seed,
            ),
            use_cache=True,
        )
        print(tokenizer.decode(output_ids[0].tolist()))
        return 0
    if args.command == "benchmark":
        document = run_benchmark(
            model_config=ModelConfig.from_toml(args.config),
            device=args.device,
            dtype_name=args.dtype,
            batch_size=args.batch_size,
            prompt_length=args.prompt_length,
            new_tokens=args.new_tokens,
            warmup=args.warmup,
            repeats=args.repeats,
            seed=args.seed,
        )
        save_benchmark(document, args.output)
        summary = {
            "output": str(args.output),
            "correctness": document["correctness"],
            "sdpa_over_manual_p50_speedup": document["results"]["sdpa_over_manual_p50_speedup"],
            "dynamic_cache_over_full_p50_speedup": document["results"][
                "dynamic_cache_over_full_p50_speedup"
            ],
            "static_cache_over_full_p50_speedup": document["results"][
                "static_cache_over_full_p50_speedup"
            ],
            "dynamic_single_token_over_full_p50_speedup": document["results"][
                "dynamic_single_token_over_full_p50_speedup"
            ],
            "static_single_token_over_full_p50_speedup": document["results"][
                "static_single_token_over_full_p50_speedup"
            ],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "external-chat":
        if not args.yes_i_understand_this_may_cost_money:
            print(
                "Refusing external request without the explicit cost acknowledgement flag.",
                file=sys.stderr,
            )
            return 2
        provider = OpenAICompatibleProvider(OpenAICompatibleConfig.from_environment())
        response = provider.complete(
            [ChatMessage(role="user", content=args.prompt)],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(response)
        return 0
    raise RuntimeError(f"unhandled command: {args.command}")
