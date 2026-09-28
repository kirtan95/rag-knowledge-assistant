"""Retrieval: cosine top-k over chunk embeddings.

Two layers:

1. :func:`rank_chunks` — pure-Python cosine ranking over in-memory
   ``(id, vector)`` pairs. No DB, no model: this is what the unit tests and
   the eval harness's offline mode exercise.
2. :func:`retrieve_from_db` — the production path: a single SQL query using
   pgvector's ``<=>`` cosine-distance operator with ``ORDER BY ... LIMIT``,
   so ranking happens inside Postgres (index-friendly) instead of pulling
   every embedding into Python.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

from sqlalchemy.orm import Session

from app.models import Chunk


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1]. Zero vectors score 0.0 (never NaN)."""
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def rank_chunks(
    query_vector: Sequence[float],
    candidates: Iterable[Mapping],
    top_k: int = 5,
) -> list[dict]:
    """Rank candidate chunks by cosine similarity to the query vector.

    Each candidate is a mapping with at least ``"id"`` and ``"embedding"``
    keys; extra keys (content, filename, ...) pass through untouched.
    Ties keep input order (Python's sort is stable).
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    scored = [
        (cosine_similarity(query_vector, c["embedding"]), c) for c in candidates
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [{**c, "score": score} for score, c in scored[:top_k]]


def retrieve_from_db(session: Session, query_vector: Sequence[float], top_k: int = 5):
    """Production retrieval: pgvector cosine distance, ranked in SQL."""
    rows = (
        session.query(Chunk, Chunk.embedding.cosine_distance(query_vector).label("distance"))
        .order_by("distance")
        .limit(top_k)
        .all()
    )
    results = []
    for chunk, distance in rows:
        results.append(
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "filename": chunk.document.filename if chunk.document else "",
                # pgvector's <=> is cosine *distance*; convert to similarity.
                "score": 1.0 - float(distance),
            }
        )
    return results
