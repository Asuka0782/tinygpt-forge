"""A deterministic character tokenizer used by the zero-dependency teaching path."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tinygpt_forge.serialization import read_json_object

TOKENIZER_FORMAT = "tinygpt-forge-character-v1"


@dataclass(frozen=True)
class CharacterTokenizer:
    """Map Unicode code points to deterministic integer IDs."""

    tokens: tuple[str, ...]
    unk_token: str = "<unk>"

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("tokens must not be empty")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("tokens must be unique")
        if self.unk_token not in self.tokens:
            raise ValueError("unk_token must be present in tokens")
        for token in self.tokens:
            if token != self.unk_token and len(token) != 1:
                raise ValueError("character vocabulary entries must be one Unicode code point")

    @classmethod
    def train(cls, text: str, *, unk_token: str = "<unk>") -> CharacterTokenizer:
        """Build a stable vocabulary sorted by Unicode code point."""

        if not text:
            raise ValueError("cannot train a tokenizer from empty text")
        characters = sorted(set(text))
        if unk_token in characters:
            raise ValueError("unk_token conflicts with a literal corpus character")
        return cls(tokens=(unk_token, *characters), unk_token=unk_token)

    @property
    def vocab_size(self) -> int:
        """Return the number of token IDs."""

        return len(self.tokens)

    @property
    def unk_id(self) -> int:
        """Return the unknown-token ID."""

        return self.tokens.index(self.unk_token)

    def encode(self, text: str, *, strict: bool = False) -> list[int]:
        """Encode text, optionally rejecting the first out-of-vocabulary character."""

        token_to_id = {token: index for index, token in enumerate(self.tokens)}
        unknown_id = self.unk_id
        encoded: list[int] = []
        for position, character in enumerate(text):
            token_id = token_to_id.get(character)
            if token_id is None:
                if strict:
                    raise ValueError(
                        f"out-of-vocabulary character at position {position}: {character!r}"
                    )
                token_id = unknown_id
            encoded.append(token_id)
        return encoded

    def decode(self, token_ids: Iterable[int]) -> str:
        """Decode IDs, rejecting negative and out-of-range values."""

        pieces: list[str] = []
        for token_id in token_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise ValueError(f"token ID must be an integer, got {token_id!r}")
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"token ID out of range: {token_id}")
            pieces.append(self.tokens[token_id])
        return "".join(pieces)

    def to_dict(self) -> dict[str, object]:
        """Return a versioned, JSON-safe representation."""

        return {
            "format": TOKENIZER_FORMAT,
            "unk_token": self.unk_token,
            "tokens": list(self.tokens),
        }

    def fingerprint(self) -> str:
        """Return a SHA-256 digest of the canonical tokenizer representation."""

        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def save(self, path: str | Path) -> None:
        """Save UTF-8 JSON without embedding local filesystem paths."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path: str | Path) -> CharacterTokenizer:
        """Load and validate a versioned tokenizer JSON file."""

        document = read_json_object(path, kind="character tokenizer")
        if document.get("format") != TOKENIZER_FORMAT:
            raise ValueError("unsupported character tokenizer format")
        tokens = document.get("tokens")
        unk_token = document.get("unk_token")
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            raise ValueError("tokenizer tokens must be a list of strings")
        if not isinstance(unk_token, str):
            raise ValueError("tokenizer unk_token must be a string")
        return cls(tokens=tuple(tokens), unk_token=unk_token)
