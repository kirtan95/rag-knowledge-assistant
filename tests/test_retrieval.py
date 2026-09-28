"""Unit tests for retrieval ranking.

All vectors are hand-made fakes: no embedding model is downloaded and no
database is needed. These tests pin the math the production pgvector query
must agree with (see eval/eval.py --mode db).
"""

import math

import pytest

from app.retrieval import cosine_similarity, rank_chunks


def test_cosine_identical_vectors_is_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors_is_minus_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_ignores_magnitude():
    # Same direction, different lengths -> still 1.0 (why cosine suits text).
    assert cosine_similarity([1.0, 1.0], [10.0, 10.0]) == pytest.approx(1.0)


def test_cosine_zero_vector_is_zero_not_nan():
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert not math.isnan(cosine_similarity([0.0], [0.0]))


def test_cosine_length_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 2.0], [1.0])


def _cand(cid, vec, **extra):
    return {"id": cid, "embedding": vec, **extra}


def test_rank_orders_by_similarity_descending():
    query = [1.0, 0.0]
    cands = [
        _cand("worst", [-1.0, 0.0]),
        _cand("best", [1.0, 0.0]),
        _cand("mid", [1.0, 1.0]),
    ]
    ranked = rank_chunks(query, cands, top_k=3)
    assert [c["id"] for c in ranked] == ["best", "mid", "worst"]
    scores = [c["score"] for c in ranked]
    assert scores == sorted(scores, reverse=True)
    assert ranked[0]["score"] == pytest.approx(1.0)


def test_rank_top_k_truncates():
    query = [1.0, 0.0, 0.0]
    cands = [_cand(f"c{i}", [float(i + 1), 0.0, 0.0]) for i in range(10)]
    ranked = rank_chunks(query, cands, top_k=4)
    assert len(ranked) == 4


def test_rank_top_k_larger_than_candidates():
    ranked = rank_chunks([1.0], [_cand("only", [1.0])], top_k=10)
    assert len(ranked) == 1


def test_rank_preserves_extra_fields():
    ranked = rank_chunks(
        [1.0, 0.0],
        [_cand("a", [1.0, 0.0], content="hello", filename="doc.md")],
        top_k=1,
    )
    assert ranked[0]["content"] == "hello"
    assert ranked[0]["filename"] == "doc.md"


def test_rank_ties_keep_input_order():
    # Python's sort is stable: equal scores keep candidate order.
    cands = [_cand("first", [1.0, 0.0]), _cand("second", [2.0, 0.0])]
    ranked = rank_chunks([1.0, 0.0], cands, top_k=2)
    assert [c["id"] for c in ranked] == ["first", "second"]


def test_rank_invalid_top_k_raises():
    with pytest.raises(ValueError):
        rank_chunks([1.0], [_cand("a", [1.0])], top_k=0)


def test_rank_matches_hand_computed_order():
    # 384-dim style smoke: ranking must agree with explicit cosine math.
    query = [0.6, 0.8, 0.0]
    cands = [
        _cand("a", [1.0, 0.0, 0.0]),   # cos = 0.6
        _cand("b", [0.0, 1.0, 0.0]),   # cos = 0.8
        _cand("c", [0.6, 0.8, 0.0]),   # cos = 1.0
    ]
    ranked = rank_chunks(query, cands, top_k=3)
    assert [c["id"] for c in ranked] == ["c", "b", "a"]
    assert ranked[1]["score"] == pytest.approx(0.8)
