"""Answer generation with Anthropic Claude (optional).

If ``ANTHROPIC_API_KEY`` is set, /query grounds a Claude response in the
retrieved chunks and cites them. If it is not set, the API does NOT fail —
it returns the retrieved context with a clear notice, because retrieval is
the valuable part of RAG and works fully offline (modulo the embedding
model download on first run).
"""

from __future__ import annotations

from app.config import settings

SYSTEM_PROMPT = (
    "You answer questions using ONLY the context chunks provided below. "
    "If the answer is not in the context, say you don't know — do not invent facts. "
    "Cite the chunks you used as [1], [2], etc."
)


def build_context_block(chunks: list[dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[{i}] (from {chunk.get('filename', 'unknown')}):\n{chunk['content']}")
    return "\n\n".join(lines)


def build_prompt(question: str, chunks: list[dict]) -> str:
    return (
        f"Context:\n{build_context_block(chunks)}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely, citing chunks like [1], [2]."
    )


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Generate a grounded answer with Claude. Raises RuntimeError on misuse."""
    api_key = settings.anthropic_api_key
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set; cannot generate an answer.")
    if not chunks:
        return "I don't know — no relevant context was retrieved."
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "The 'anthropic' package is not installed, but ANTHROPIC_API_KEY is set."
        ) from exc

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
    )
    return "".join(block.text for block in message.content if hasattr(block, "text"))


NO_KEY_NOTICE = (
    "ANTHROPIC_API_KEY is not set, so no LLM answer was generated. "
    "The retrieved context below is the raw material the answer would be grounded in — "
    "set the key to get a synthesized, cited answer."
)
