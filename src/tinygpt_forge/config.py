"""Validated model configuration and TOML loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from tinygpt_forge.toml_compat import load_toml

ATTENTION_BACKENDS = frozenset({"manual", "sdpa"})


@dataclass(frozen=True)
class ModelConfig:
    """Semantic configuration shared by the teaching and optimized backends."""

    vocab_size: int
    max_seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    rope_base: float = 10_000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    attention_backend: str = "sdpa"

    def __post_init__(self) -> None:
        positive_ints = {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "d_ff": self.d_ff,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for GQA")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for pairwise RoPE")
        if self.rope_base <= 0:
            raise ValueError("rope_base must be positive")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if self.attention_backend not in ATTENTION_BACKENDS:
            choices = ", ".join(sorted(ATTENTION_BACKENDS))
            raise ValueError(f"attention_backend must be one of: {choices}")

    @property
    def head_dim(self) -> int:
        """Return the dimension of each query head."""

        return self.d_model // self.n_heads

    @property
    def query_groups(self) -> int:
        """Return the number of query heads that share one key/value head."""

        return self.n_heads // self.n_kv_heads

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration mapping."""

        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ModelConfig:
        """Build a config and reject unknown keys instead of silently ignoring typos."""

        known = {field.name for field in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown model config fields: {names}")
        return cls(**dict(raw))

    @classmethod
    def from_toml(cls, path: str | Path) -> ModelConfig:
        """Load the `[model]` table from a TOML file."""

        config_path = Path(path)
        document = load_toml(config_path)
        raw = document.get("model")
        if not isinstance(raw, Mapping):
            raise ValueError(f"{config_path} must contain a [model] table")
        return cls.from_mapping(raw)
