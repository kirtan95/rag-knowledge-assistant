"""Embedding service: sentence-transformers, lazy-loaded and cached.

The model (default ``all-MiniLM-L6-v2``) is loaded exactly once per process
on first use and reused afterwards. Loading is lazy — not at import time —
so unit tests, the eval harness's non-embedding paths, and ``--help`` never
pay the multi-hundred-MB download/load cost or require network access.

Thread-safety note: model loading is guarded by a lock; ``encode`` itself is
safe to call from FastAPI's threadpool because sentence-transformers
releases the GIL in the underlying torch ops.
"""

from __future__ import annotations

import threading

from app.config import settings

_lock = threading.Lock()
_model = None
_model_name_loaded: str | None = None


def get_embedder(model_name: str | None = None):
    """Return the cached SentenceTransformer, loading it on first call."""
    global _model, _model_name_loaded
    name = model_name or settings.embedding_model
    if _model is None or _model_name_loaded != name:
        with _lock:
            if _model is None or _model_name_loaded != name:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(name)
                _model_name_loaded = name
    return _model


def embedding_dim(model_name: str | None = None) -> int:
    return get_embedder(model_name).get_sentence_embedding_dimension()


def embed_texts(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    """Embed a batch of texts. Returns a list of float vectors, one per text."""
    if not texts:
        return []
    model = get_embedder(model_name)
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [vec.astype(float).tolist() for vec in vectors]


def embed_text(text: str, model_name: str | None = None) -> list[float]:
    return embed_texts([text], model_name)[0]
