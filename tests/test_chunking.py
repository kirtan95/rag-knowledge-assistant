"""Unit tests for the chunker.

No model download, no DB, no network: both the tiktoken path and the
word-fallback path are exercised. tiktoken-specific tests are skipped if
tiktoken is not installed.
"""

import pytest

from app.chunking import chunk_text, chunk_token_spans, count_tokens, decode, encode

tiktoken = pytest.importorskip("tiktoken", reason="tiktoken not installed")


LOREM = (
    "Retrieval augmented generation grounds a language model in your own documents. "
    "First the documents are split into chunks. Then each chunk is embedded into a vector. "
    "At query time the question is embedded the same way. "
    "The most similar chunks are retrieved by cosine similarity. "
    "Finally the language model writes an answer using only the retrieved context. "
    "This reduces hallucination because every claim can cite its source. "
)


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ", chunk_size=10, chunk_overlap=2) == []


def test_short_text_is_single_chunk():
    chunks = chunk_text("Hello world.", chunk_size=500, chunk_overlap=50)
    assert chunks == ["Hello world."]


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=50, chunk_overlap=50)
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=50, chunk_overlap=60)
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=0)


def test_overlap_is_exact_when_boundaries_disabled():
    # word-fallback: token == word, so overlap assertions are exact.
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(
        text, chunk_size=20, chunk_overlap=5,
        use_tiktoken=False, respect_sentence_boundaries=False,
    )
    assert len(chunks) > 2
    for prev, nxt in zip(chunks, chunks[1:]):
        prev_toks = prev.split()
        nxt_toks = nxt.split()
        assert nxt_toks[:5] == prev_toks[-5:], "next chunk must repeat the overlap window"


def test_overlap_spans_are_exact_tiktoken():
    # Tokenizer-native invariant: the next window starts exactly
    # `overlap` tokens before the previous one ended. (Re-encoding decoded
    # chunk text is NOT used: tiktoken merges tokens based on surrounding
    # context, so a chunk re-encodes differently in isolation.)
    text = " ".join(f"word{i}" for i in range(200))
    toks = encode(text, use_tiktoken=True)
    spans = chunk_token_spans(toks, 30, 7, use_tiktoken=True,
                              respect_sentence_boundaries=False)
    assert len(spans) > 2
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert s2 == e1 - 7, "overlap window must be exactly `overlap` tokens"


def test_adjacent_chunks_share_overlap_text():
    # The shared token slice decodes to text that is a prefix of the next
    # chunk and a suffix of the previous one.
    text = " ".join(f"word{i}" for i in range(200))
    for use_tiktoken in (True, False):
        toks = encode(text, use_tiktoken=use_tiktoken)
        spans = chunk_token_spans(toks, 30, 7, use_tiktoken=use_tiktoken,
                                  respect_sentence_boundaries=False)
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            overlap_text = decode(toks[s2:e1], use_tiktoken)
            assert decode(toks[s2:e2], use_tiktoken).startswith(overlap_text)
            assert decode(toks[s1:e1], use_tiktoken).endswith(overlap_text)


def test_chunks_respect_size_budget():
    text = LOREM * 20
    chunks = chunk_text(text, chunk_size=40, chunk_overlap=8, use_tiktoken=False,
                        respect_sentence_boundaries=False)
    for c in chunks:
        assert len(c.split()) <= 40


def test_sentence_boundary_snapping():
    # Long single sentences that would be cut mid-sentence without snapping.
    sentences = [
        "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. ",
        "Nu xi omicron pi rho sigma tau upsilon phi chi psi omega. ",
        "The quick brown fox jumps over the lazy dog near the river bank. ",
    ]
    text = "".join(sentences * 6)
    chunks = chunk_text(text, chunk_size=25, chunk_overlap=5,
                        use_tiktoken=False, respect_sentence_boundaries=True)
    assert len(chunks) > 1
    for c in chunks[:-1]:
        assert c.rstrip().endswith((".", "!", "?")), f"chunk should end on a sentence: {c!r}"


def test_no_empty_chunks_and_full_coverage():
    text = LOREM * 10
    chunks = chunk_text(text, chunk_size=30, chunk_overlap=6, use_tiktoken=False,
                        respect_sentence_boundaries=False)
    assert all(c.strip() for c in chunks)
    # every word of the input appears in at least one chunk
    covered = set(w for c in chunks for w in c.split())
    assert set(text.split()) <= covered


def test_deterministic():
    text = LOREM * 5
    kwargs = dict(chunk_size=33, chunk_overlap=7, use_tiktoken=False)
    assert chunk_text(text, **kwargs) == chunk_text(text, **kwargs)


def test_count_tokens_fallback_counts_words():
    assert count_tokens("one two three", use_tiktoken=False) == 3


def test_count_tokens_tiktoken_positive():
    assert count_tokens("hello world", use_tiktoken=True) > 0


def test_encode_decode_roundtrip_tiktoken():
    toks = encode("Hello, world!", use_tiktoken=True)
    assert decode(toks, use_tiktoken=True) == "Hello, world!"


def test_encode_decode_roundtrip_fallback():
    toks = encode("Hello world", use_tiktoken=False)
    assert toks == ["Hello", "world"]
    assert decode(toks, use_tiktoken=False) == "Hello world"
