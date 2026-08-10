"""Safe, weights-only checkpoints with explicit model configuration."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, load_model, save_file, save_model
from torch.amp.grad_scaler import GradScaler

from tinygpt_forge.config import ModelConfig
from tinygpt_forge.model.gpt import TinyGPT
from tinygpt_forge.serialization import read_json_object

CHECKPOINT_FORMAT = "tinygpt-forge-weights-v1"
TRAINING_CHECKPOINT_FORMAT = "tinygpt-forge-training-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_weights(model: TinyGPT, directory: str | Path) -> dict[str, Any]:
    """Atomically save model weights as safetensors plus a JSON manifest."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    weights_path = destination / "model.safetensors"
    weights_temp = destination / "model.safetensors.tmp"
    manifest_path = destination / "model.json"
    manifest_temp = destination / "model.json.tmp"

    save_model(
        model,
        str(weights_temp),
        metadata={"format": CHECKPOINT_FORMAT, "torch_version": torch.__version__},
    )
    os.replace(weights_temp, weights_path)
    manifest: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "model_config": model.config.to_dict(),
        "parameter_count": model.parameter_count(),
        "weights_file": weights_path.name,
        "weights_sha256": _sha256(weights_path),
    }
    manifest_temp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(manifest_temp, manifest_path)
    return manifest


def load_weights(
    directory: str | Path,
    *,
    device: torch.device | str = "cpu",
    verify_hash: bool = True,
) -> TinyGPT:
    """Validate a manifest, instantiate its model, and strictly load safetensors weights."""

    source = Path(directory)
    manifest = read_json_object(source / "model.json", kind="model manifest")
    if manifest.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported TinyGPT Forge checkpoint format")
    raw_config = manifest.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint model_config must be an object")
    weights_name = manifest.get("weights_file")
    if not isinstance(weights_name, str) or Path(weights_name).name != weights_name:
        raise ValueError("checkpoint weights_file must be a local filename")
    weights_path = source / weights_name
    expected_hash = manifest.get("weights_sha256")
    if verify_hash and (
        not isinstance(expected_hash, str) or _sha256(weights_path) != expected_hash
    ):
        raise ValueError("checkpoint weights SHA-256 mismatch")

    config = ModelConfig.from_mapping(raw_config)
    model = TinyGPT(config)
    target_device = torch.device(device)
    model.to(target_device)
    missing, unexpected = load_model(model, weights_path, strict=True, device=str(target_device))
    if missing or unexpected:
        raise RuntimeError(f"checkpoint key mismatch: missing={missing}, unexpected={unexpected}")
    return model


def _encode_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_json_value(item) for item in value]}
    if isinstance(value, list):
        return [_encode_json_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("checkpoint JSON object keys must be strings")
        return {key: _encode_json_value(item) for key, item in value.items()}
    raise TypeError(f"value is not JSON-serializable by the checkpoint codec: {type(value)}")


def _decode_json_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_json_value(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__tuple__"}:
            items = value["__tuple__"]
            if not isinstance(items, list):
                raise ValueError("invalid tuple encoding")
            return tuple(_decode_json_value(item) for item in items)
        return {key: _decode_json_value(item) for key, item in value.items()}
    return value


def _save_optimizer(optimizer: torch.optim.Optimizer, directory: Path) -> dict[str, Any]:
    state_dict = optimizer.state_dict()
    tensors: dict[str, torch.Tensor] = {}
    scalar_state: dict[str, dict[str, Any]] = {}
    for parameter_id, parameter_state in state_dict["state"].items():
        encoded_scalars: dict[str, Any] = {}
        for name, value in parameter_state.items():
            if "/" in name:
                raise ValueError(f"optimizer state name cannot contain '/': {name}")
            if isinstance(value, torch.Tensor):
                tensors[f"state/{parameter_id}/{name}"] = value.detach().contiguous()
            else:
                encoded_scalars[name] = _encode_json_value(value)
        if encoded_scalars:
            scalar_state[str(parameter_id)] = encoded_scalars

    tensor_path = directory / "optimizer.safetensors"
    tensor_temp = directory / "optimizer.safetensors.tmp"
    save_file(
        tensors,
        tensor_temp,
        metadata={"format": TRAINING_CHECKPOINT_FORMAT},
    )
    os.replace(tensor_temp, tensor_path)
    document: dict[str, Any] = {
        "format": TRAINING_CHECKPOINT_FORMAT,
        "tensor_file": tensor_path.name,
        "tensor_sha256": _sha256(tensor_path),
        "scalar_state": scalar_state,
        "param_groups": _encode_json_value(state_dict["param_groups"]),
    }
    optimizer_path = directory / "optimizer.json"
    optimizer_temp = directory / "optimizer.json.tmp"
    optimizer_temp.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(optimizer_temp, optimizer_path)
    return document


def _read_optimizer_state(
    directory: Path,
    *,
    device: torch.device,
    expected_bundle_hash: str,
) -> dict[str, Any]:
    document = read_json_object(directory / "optimizer.json", kind="optimizer manifest")
    if document.get("format") != TRAINING_CHECKPOINT_FORMAT:
        raise ValueError("unsupported optimizer checkpoint format")
    tensor_name = document.get("tensor_file")
    if not isinstance(tensor_name, str) or Path(tensor_name).name != tensor_name:
        raise ValueError("optimizer tensor_file must be a local filename")
    tensor_path = directory / tensor_name
    expected_hash = document.get("tensor_sha256")
    if not isinstance(expected_hash, str) or _sha256(tensor_path) != expected_hash:
        raise ValueError("optimizer tensor SHA-256 mismatch")
    if expected_hash != expected_bundle_hash:
        raise ValueError("training/optimizer checkpoint hash mismatch")
    tensors = load_file(tensor_path, device=str(device))

    state: dict[int, dict[str, Any]] = {}
    for key, tensor in tensors.items():
        parts = key.split("/", maxsplit=2)
        if len(parts) != 3 or parts[0] != "state":
            raise ValueError(f"invalid optimizer tensor key: {key}")
        parameter_id = int(parts[1])
        state.setdefault(parameter_id, {})[parts[2]] = tensor
    scalar_state = document.get("scalar_state")
    if not isinstance(scalar_state, dict):
        raise ValueError("optimizer scalar_state must be an object")
    for parameter_id_text, values in scalar_state.items():
        if not isinstance(values, dict):
            raise ValueError("optimizer scalar parameter state must be an object")
        parameter_id = int(parameter_id_text)
        state.setdefault(parameter_id, {}).update(_decode_json_value(values))
    param_groups = _decode_json_value(document.get("param_groups"))
    if not isinstance(param_groups, list):
        raise ValueError("optimizer param_groups must decode to a list")
    return {"state": state, "param_groups": param_groups}


def _save_rng_states(
    batcher_rng_state: torch.Tensor,
    directory: Path,
) -> dict[str, Any]:
    tensors = {
        "torch_cpu": torch.get_rng_state().contiguous(),
        "batcher_cpu": batcher_rng_state.cpu().contiguous(),
    }
    for index, state in enumerate(
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    ):
        tensors[f"torch_cuda_{index}"] = state.cpu().contiguous()
    rng_path = directory / "rng.safetensors"
    rng_temp = directory / "rng.safetensors.tmp"
    save_file(tensors, rng_temp, metadata={"format": TRAINING_CHECKPOINT_FORMAT})
    os.replace(rng_temp, rng_path)
    return {
        "file": rng_path.name,
        "sha256": _sha256(rng_path),
        "cuda_device_count": sum(name.startswith("torch_cuda_") for name in tensors),
    }


def _read_rng_states(
    directory: Path,
    document: Mapping[str, Any],
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    filename = document.get("file")
    expected_hash = document.get("sha256")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("RNG file must be a local filename")
    path = directory / filename
    if not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
        raise ValueError("RNG state SHA-256 mismatch")
    tensors = load_file(path, device="cpu")
    if "torch_cpu" not in tensors or "batcher_cpu" not in tensors:
        raise ValueError("RNG checkpoint is missing required CPU states")
    raw_cuda_count = document.get("cuda_device_count")
    if (
        not isinstance(raw_cuda_count, int)
        or isinstance(raw_cuda_count, bool)
        or raw_cuda_count < 0
    ):
        raise ValueError("RNG CUDA device count must be a non-negative integer")
    cuda_states = [tensors[f"torch_cuda_{index}"] for index in range(raw_cuda_count)]
    if len(cuda_states) != torch.cuda.device_count():
        raise ValueError("CUDA RNG device count changed since checkpoint creation")
    return tensors["torch_cpu"], cuda_states, tensors["batcher_cpu"]


def save_training_checkpoint(
    *,
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    batcher_rng_state: torch.Tensor,
    scaler: GradScaler,
    trainer_state: Mapping[str, Any],
    directory: str | Path,
) -> dict[str, Any]:
    """Save a pickle-free training continuation checkpoint."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    model_manifest = save_weights(model, destination)
    optimizer_manifest = _save_optimizer(optimizer, destination)
    rng_manifest = _save_rng_states(batcher_rng_state, destination)
    document: dict[str, Any] = {
        "format": TRAINING_CHECKPOINT_FORMAT,
        "model_weights_sha256": model_manifest["weights_sha256"],
        "optimizer_tensors_sha256": optimizer_manifest["tensor_sha256"],
        "rng": rng_manifest,
        "scaler_state": _encode_json_value(scaler.state_dict()),
        "trainer_state": _encode_json_value(dict(trainer_state)),
    }
    manifest_path = destination / "training.json"
    manifest_temp = destination / "training.json.tmp"
    manifest_temp.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(manifest_temp, manifest_path)
    return document


def load_training_checkpoint(
    *,
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    directory: str | Path,
    device: torch.device | str,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Strictly restore model, optimizer, scaler, global RNG, and batcher RNG."""

    source = Path(directory)
    document = read_json_object(source / "training.json", kind="training manifest")
    if document.get("format") != TRAINING_CHECKPOINT_FORMAT:
        raise ValueError("unsupported training checkpoint format")
    model_manifest = read_json_object(source / "model.json", kind="model manifest")
    if model_manifest.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported model checkpoint format in training bundle")
    if model_manifest.get("model_config") != model.config.to_dict():
        raise ValueError("training checkpoint model configuration mismatch")
    weights_name = model_manifest.get("weights_file")
    if not isinstance(weights_name, str) or Path(weights_name).name != weights_name:
        raise ValueError("training checkpoint weights_file must be a local filename")
    weights_path = source / weights_name
    expected_hash = document.get("model_weights_sha256")
    if not isinstance(expected_hash, str) or _sha256(weights_path) != expected_hash:
        raise ValueError("training checkpoint model SHA-256 mismatch")
    if model_manifest.get("weights_sha256") != expected_hash:
        raise ValueError("training/model checkpoint hash mismatch")
    target_device = torch.device(device)
    optimizer_hash = document.get("optimizer_tensors_sha256")
    if not isinstance(optimizer_hash, str):
        raise ValueError("training checkpoint optimizer hash is missing")
    optimizer_state = _read_optimizer_state(
        source,
        device=target_device,
        expected_bundle_hash=optimizer_hash,
    )
    scaler_state = _decode_json_value(document.get("scaler_state"))
    if not isinstance(scaler_state, dict):
        raise ValueError("scaler_state must decode to an object")
    rng_document = document.get("rng")
    if not isinstance(rng_document, dict):
        raise ValueError("RNG manifest must be an object")
    cpu_rng_state, cuda_rng_states, batcher_state = _read_rng_states(source, rng_document)
    trainer_state = _decode_json_value(document.get("trainer_state"))
    if not isinstance(trainer_state, dict):
        raise ValueError("trainer_state must decode to an object")

    missing, unexpected = load_model(model, weights_path, strict=True, device=str(target_device))
    if missing or unexpected:
        raise RuntimeError(f"checkpoint key mismatch: missing={missing}, unexpected={unexpected}")
    optimizer.load_state_dict(optimizer_state)
    scaler.load_state_dict(scaler_state)
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state_all(cuda_rng_states)
    return trainer_state, batcher_state
