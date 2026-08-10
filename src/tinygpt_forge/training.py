"""A small, reproducible character-language-model training loop."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.amp.grad_scaler import GradScaler

from tinygpt_forge.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
    save_weights,
)
from tinygpt_forge.config import ModelConfig
from tinygpt_forge.data import NextTokenBatcher, split_text
from tinygpt_forge.model.gpt import TinyGPT
from tinygpt_forge.tokenizer import CharacterTokenizer
from tinygpt_forge.toml_compat import load_toml


@dataclass(frozen=True)
class TrainingConfig:
    """Controls for a deterministic single-process baseline run."""

    batch_size: int = 8
    block_size: int = 64
    max_steps: int = 100
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 20
    eval_batches: int = 10
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 42
    device: str = "auto"
    dtype: str = "float32"
    fused_optimizer: bool = True

    def __post_init__(self) -> None:
        positive_ints = {
            "batch_size": self.batch_size,
            "block_size": self.block_size,
            "max_steps": self.max_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "eval_interval": self.eval_interval,
            "eval_batches": self.eval_batches,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError(
                "learning_rate/grad_clip must be positive and weight_decay non-negative"
            )
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("AdamW beta values must satisfy 0 <= beta < 1")
        if self.validation_fraction <= 0 or self.test_fraction <= 0:
            raise ValueError("baseline training requires non-empty validation and test splits")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation_fraction + test_fraction must be less than one")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("dtype must be float32, bfloat16, or float16")

    def resume_contract(self) -> dict[str, Any]:
        """Return fields that must stay fixed when extending `max_steps`."""

        contract = asdict(self)
        contract.pop("max_steps")
        return contract

    @classmethod
    def from_toml(cls, path: str | Path) -> TrainingConfig:
        """Load and strictly validate the `[training]` TOML table."""

        document = load_toml(path)
        raw = document.get("training")
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path} must contain a [training] table")
        known = {field.name for field in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown training config fields: {', '.join(sorted(unknown))}")
        return cls(**dict(raw))


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _autocast_context(device: torch.device, dtype_name: str) -> AbstractContextManager[Any]:
    if dtype_name == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 autocast is not supported by this CPU baseline")
    return torch.autocast(device_type=device.type, dtype=dtype)


def _perplexity(loss: float) -> float:
    return math.exp(loss) if loss < 100 else math.inf


@torch.inference_mode()
def evaluate(
    model: TinyGPT,
    batcher: NextTokenBatcher,
    *,
    batches: int,
    dtype_name: str,
    backend: str,
    fixed_rng_state: Tensor,
) -> float:
    """Evaluate on the same fixed random windows every time."""

    was_training = model.training
    model.eval()
    batcher.set_rng_state(fixed_rng_state)
    losses: list[float] = []
    for _ in range(batches):
        inputs, targets = batcher.next_batch()
        with _autocast_context(inputs.device, dtype_name):
            output = model(inputs, targets=targets, backend=backend)
        if output.loss is None:
            raise RuntimeError("evaluation loss was not computed")
        losses.append(output.loss.float().item())
    model.train(was_training)
    return sum(losses) / len(losses)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def train_character_model(
    *,
    corpus_path: str | Path,
    run_directory: str | Path,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Train a local character model and write evidence-bearing run artifacts."""

    started = datetime.now(timezone.utc)
    source = Path(corpus_path)
    destination = Path(run_directory)
    if resume_from is None and destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"run directory is not empty: {destination}")
    if resume_from is not None and not destination.is_dir():
        raise FileNotFoundError("resume requires an existing run directory")
    destination.mkdir(parents=True, exist_ok=True)

    text = source.read_text(encoding="utf-8")
    source_sha256 = _file_sha256(source)
    raw_splits = split_text(
        text,
        validation_fraction=training_config.validation_fraction,
        test_fraction=training_config.test_fraction,
    )
    fitted_tokenizer = CharacterTokenizer.train(raw_splits.train)
    tokenizer_path = destination / "tokenizer.json"
    if resume_from is None:
        tokenizer = fitted_tokenizer
        tokenizer.save(tokenizer_path)
    else:
        tokenizer = CharacterTokenizer.load(tokenizer_path)
        if tokenizer.fingerprint() != fitted_tokenizer.fingerprint():
            raise ValueError("training corpus/tokenizer changed since the resume checkpoint")
    train_ids = torch.tensor(tokenizer.encode(raw_splits.train, strict=True), dtype=torch.long)
    validation_ids = torch.tensor(tokenizer.encode(raw_splits.validation), dtype=torch.long)
    test_ids = torch.tensor(tokenizer.encode(raw_splits.test), dtype=torch.long)

    for split_name, token_ids in {
        "train": train_ids,
        "validation": validation_ids,
        "test": test_ids,
    }.items():
        if token_ids.numel() <= training_config.block_size:
            raise ValueError(
                f"{split_name} split has {token_ids.numel()} tokens, which must exceed "
                f"block_size={training_config.block_size}"
            )

    device = _resolve_device(training_config.device)
    if training_config.block_size > model_config.max_seq_len:
        raise ValueError("training block_size exceeds model max_seq_len")
    resolved_model_config = replace(model_config, vocab_size=tokenizer.vocab_size)

    torch.manual_seed(training_config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training_config.seed)
    model = TinyGPT(resolved_model_config).to(device)
    fused = training_config.fused_optimizer and device.type == "cuda"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=(training_config.beta1, training_config.beta2),
        weight_decay=training_config.weight_decay,
        fused=fused,
    )
    use_scaler = device.type == "cuda" and training_config.dtype == "float16"
    scaler = GradScaler(device.type, enabled=use_scaler)

    train_batcher = NextTokenBatcher(
        train_ids,
        block_size=training_config.block_size,
        batch_size=training_config.batch_size,
        seed=training_config.seed + 1,
        device=device,
    )
    validation_batcher = NextTokenBatcher(
        validation_ids,
        block_size=training_config.block_size,
        batch_size=training_config.batch_size,
        seed=training_config.seed + 2,
        device=device,
    )
    test_batcher = NextTokenBatcher(
        test_ids,
        block_size=training_config.block_size,
        batch_size=training_config.batch_size,
        seed=training_config.seed + 3,
        device=device,
    )
    validation_state = validation_batcher.get_rng_state()
    test_state = test_batcher.get_rng_state()

    checkpoints = destination / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_validation_loss = math.inf
    best_step = 0
    start_step = 0
    original_started_utc = started.isoformat()
    resumed_from_step: int | None = None

    if resume_from is not None:
        trainer_state, train_batcher_state = load_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            directory=resume_from,
            device=device,
        )
        if trainer_state.get("source_sha256") != source_sha256:
            raise ValueError("training corpus hash changed since the resume checkpoint")
        if trainer_state.get("tokenizer_fingerprint") != tokenizer.fingerprint():
            raise ValueError("tokenizer fingerprint changed since the resume checkpoint")
        if trainer_state.get("training_contract") != training_config.resume_contract():
            raise ValueError(
                "training configuration changed outside the allowed max_steps extension"
            )
        raw_step = trainer_state.get("step")
        raw_best_step = trainer_state.get("best_step")
        raw_best_loss = trainer_state.get("best_validation_loss")
        raw_history = trainer_state.get("history")
        raw_started = trainer_state.get("started_utc")
        if (
            not isinstance(raw_step, int)
            or isinstance(raw_step, bool)
            or not isinstance(raw_best_step, int)
            or isinstance(raw_best_step, bool)
            or not isinstance(raw_best_loss, (int, float))
            or isinstance(raw_best_loss, bool)
            or not isinstance(raw_history, list)
            or not all(isinstance(record, dict) for record in raw_history)
            or not isinstance(raw_started, str)
        ):
            raise ValueError("training checkpoint trainer_state has invalid field types")
        if (
            raw_step < 0
            or not 0 <= raw_best_step <= raw_step
            or not math.isfinite(float(raw_best_loss))
        ):
            raise ValueError("training checkpoint trainer_state has invalid field values")
        start_step = raw_step
        resumed_from_step = start_step
        best_step = raw_best_step
        best_validation_loss = float(raw_best_loss)
        history = raw_history
        original_started_utc = raw_started
        train_batcher.set_rng_state(train_batcher_state)
        if training_config.max_steps <= start_step:
            raise ValueError("resumed max_steps must be greater than the checkpoint step")

    def run_evaluation(step: int, train_loss: float | None) -> None:
        nonlocal best_validation_loss, best_step
        validation_loss = evaluate(
            model,
            validation_batcher,
            batches=training_config.eval_batches,
            dtype_name=training_config.dtype,
            backend=resolved_model_config.attention_backend,
            fixed_rng_state=validation_state,
        )
        record: dict[str, float | int] = {
            "step": step,
            "validation_loss": validation_loss,
            "validation_perplexity": _perplexity(validation_loss),
        }
        if train_loss is not None:
            record["train_loss"] = train_loss
        history.append(record)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_step = step
            save_weights(model, checkpoints / "best")

        if step > 0:
            save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                batcher_rng_state=train_batcher.get_rng_state(),
                scaler=scaler,
                trainer_state={
                    "step": step,
                    "best_step": best_step,
                    "best_validation_loss": best_validation_loss,
                    "history": history,
                    "source_sha256": source_sha256,
                    "tokenizer_fingerprint": tokenizer.fingerprint(),
                    "training_contract": training_config.resume_contract(),
                    "started_utc": original_started_utc,
                },
                directory=checkpoints / "resume",
            )

    if resume_from is None:
        run_evaluation(0, None)
    model.train()
    for step in range(start_step + 1, training_config.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(training_config.gradient_accumulation_steps):
            inputs, targets = train_batcher.next_batch()
            with _autocast_context(device, training_config.dtype):
                output = model(
                    inputs,
                    targets=targets,
                    backend=resolved_model_config.attention_backend,
                )
                if output.loss is None:
                    raise RuntimeError("training loss was not computed")
                micro_loss = output.loss / training_config.gradient_accumulation_steps
            accumulated_loss += output.loss.detach().float().item()
            scaler.scale(micro_loss).backward()

        if use_scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        mean_train_loss = accumulated_loss / training_config.gradient_accumulation_steps

        if step % training_config.eval_interval == 0 or step == training_config.max_steps:
            run_evaluation(step, mean_train_loss)
            model.train()

    save_weights(model, checkpoints / "last")
    test_loss = evaluate(
        model,
        test_batcher,
        batches=training_config.eval_batches,
        dtype_name=training_config.dtype,
        backend=resolved_model_config.attention_backend,
        fixed_rng_state=test_state,
    )
    finished = datetime.now(timezone.utc)
    result: dict[str, Any] = {
        "format": "tinygpt-forge-run-v1",
        "status": "completed",
        "started_utc": original_started_utc,
        "finished_utc": finished.isoformat(),
        "duration_seconds_this_invocation": (finished - started).total_seconds(),
        "resumed_from_step": resumed_from_step,
        "source": {"name": source.name, "sha256": source_sha256},
        "tokenizer": {
            "format": "character",
            "vocab_size": tokenizer.vocab_size,
            "fingerprint": tokenizer.fingerprint(),
            "validation_oov": int((validation_ids == tokenizer.unk_id).sum().item()),
            "test_oov": int((test_ids == tokenizer.unk_id).sum().item()),
        },
        "split_tokens": {
            "train": train_ids.numel(),
            "validation": validation_ids.numel(),
            "test": test_ids.numel(),
        },
        "model_config": resolved_model_config.to_dict(),
        "training_config": asdict(training_config),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "fused_optimizer_effective": fused,
        },
        "parameter_count": model.parameter_count(),
        "best_step": best_step,
        "best_validation_loss": best_validation_loss,
        "best_validation_perplexity": _perplexity(best_validation_loss),
        "test_loss_last": test_loss,
        "test_perplexity_last": _perplexity(test_loss),
        "history": history,
    }
    _write_json_atomic(destination / "run.json", result)
    return result
