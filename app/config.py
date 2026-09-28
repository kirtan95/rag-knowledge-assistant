"""Centralized, env-driven configuration.

Every tunable lives here so the API code never reads os.environ directly.
See .env.example for the full list of variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    database_url: str = field(
        default_factory=lambda: os.environ.get(
            "DATABASE_URL", "postgresql+psycopg://rag:ragpassword@localhost:5432/ragdb"
        )
    )
    anthropic_api_key: str | None = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY") or None
    )
    anthropic_model: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    )
    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 500))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 50))
    top_k: int = field(default_factory=lambda: _int("TOP_K", 5))
    embedding_model: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    embedding_dim: int = 384  # output dim of all-MiniLM-L6-v2
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MiB per upload

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than "
                f"CHUNK_SIZE ({self.chunk_size})"
            )


settings = Settings()
