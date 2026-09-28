"""Token-aware sliding-window chunker.

Why this design (the short version; the full rationale is in the README):

* Tokens, not characters: chunk budgets exist because embedding models and
  LLMs bill/think in tokens. A "500-character" chunk of dense prose can be
  700 tokens in one language and 300 in another; token budgets are stable.
* tiktoken (``cl100k_base``) when available because it is the de-facto
  standard tokenizer and dependency-free at runtime. If tiktoken is not
  installed we fall back to whitespace word counts so the pipeline still
  works (e.g. minimal containers, offline installs). The fallback is
  documented as approximate, never silent-magic.
* Sliding window with overlap: a hard cut can strand half a sentence (or a
  key fact) at a boundary. Overlap keeps the tail of chunk N at the head of
  chunk N+1, so a fact near a boundary is still retrievable in full context.
* Sentence-boundary snapping: within the last 40% of each window we snap the
  cut to the last sentence-ending token, so chunks end on sentence
  boundaries instead of mid-word. This is a heuristic (it can be fooled by
  abbreviations); it is a strict improvement over arbitrary cuts and can be
  disabled per-call.
"""

from __future__ import annotations

from functools import lru_cache

SENTENCE_ENDINGS = (".", "!", "?")


@lru_cache(maxsize=1)
def _tiktoken_encoding():
    """Return the cl100k_base encoding, or None if tiktoken is unavailable."""
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def tokenization_mode() -> str:
    """Human-readable name of the active tokenizer (for logs / README demos)."""
    return "tiktoken(cl100k_base)" if _tiktoken_encoding() is not None else "word-fallback"


def encode(text: str, use_tiktoken: bool = True) -> list:
    """Encode text to a token list. Tokens are ints (tiktoken) or words (fallback)."""
    enc = _tiktoken_encoding() if use_tiktoken else None
    if enc is not None:
        return enc.encode(text)
    return text.split()


def decode(tokens: list, use_tiktoken: bool = True) -> str:
    enc = _tiktoken_encoding() if use_tiktoken else None
    if enc is not None:
        return enc.decode(tokens)
    return " ".join(tokens)


def count_tokens(text: str, use_tiktoken: bool = True) -> int:
    return len(encode(text, use_tiktoken))


def _find_sentence_boundary(tokens: list, lo: int, hi: int, use_tiktoken: bool) -> int | None:
    """Scan tokens[lo:hi] right-to-left for a sentence-ending token.

    Returns the cut index (exclusive end) just after that token, or None.
    Single-character punctuation tokens (".", "!", "?") are ignored to avoid
    snapping on abbreviations like "e.g."; a token must carry real content.
    """
    enc = _tiktoken_encoding() if use_tiktoken else None
    for i in range(hi - 1, lo - 1, -1):
        tok = tokens[i]
        piece = enc.decode([tok]) if enc is not None else tok
        stripped = piece.strip()
        if len(stripped) > 1 and stripped.endswith(SENTENCE_ENDINGS):
            return i + 1
    return None


def chunk_token_spans(
    tokens: list,
    chunk_size: int,
    chunk_overlap: int,
    use_tiktoken: bool = True,
    respect_sentence_boundaries: bool = True,
) -> list[tuple[int, int]]:
    """Return (start, end) token spans for the sliding window.

    Separated from :func:`chunk_text` so the overlap invariant
    (``spans[i+1].start == spans[i].end - chunk_overlap``) is directly testable.
    """
    n = len(tokens)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n and respect_sentence_boundaries:
            # Only search the tail of the window: snapping too early would
            # produce many tiny chunks and defeat the size budget.
            search_lo = start + int(chunk_size * 0.6)
            boundary = _find_sentence_boundary(tokens, search_lo, end, use_tiktoken)
            if boundary is not None and boundary > start:
                end = boundary
        spans.append((start, end))
        if end >= n:
            break
        # Guarantee forward progress even in degenerate cases.
        start = max(end - chunk_overlap, start + 1)
    return spans


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    use_tiktoken: bool = True,
    respect_sentence_boundaries: bool = True,
) -> list[str]:
    """Split text into overlapping token windows.

    Args:
        text: input document text.
        chunk_size: target tokens per chunk.
        chunk_overlap: tokens repeated at the start of the next chunk.
        use_tiktoken: use tiktoken if installed; otherwise word fallback.
        respect_sentence_boundaries: snap cuts to sentence ends (heuristic).

    Returns:
        List of chunk strings, in document order, with no empty chunks.

    Raises:
        ValueError: if chunk_overlap >= chunk_size or sizes are not positive.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    tokens = encode(text, use_tiktoken)
    if not tokens:
        return []

    chunks: list[str] = []
    for start, end in chunk_token_spans(
        tokens, chunk_size, chunk_overlap, use_tiktoken, respect_sentence_boundaries
    ):
        piece = decode(tokens[start:end], use_tiktoken).strip()
        if piece:
            chunks.append(piece)
    return chunks
