"""End-to-end ingest -> query logic with fake vectors.

Simulates the full RAG loop — chunk documents, "embed" chunks and question,
rank by cosine similarity — using a deterministic bag-of-words fake embedder
instead of sentence-transformers. No model download, no database.
"""

from app.chunking import chunk_text
from app.retrieval import rank_chunks

DOCS = {
    "deploy.md": (
        "To deploy, run docker compose up --build. "
        "The compose file starts Postgres with pgvector and the FastAPI api. "
        "Copy .env.example to .env before the first start."
    ),
    "chunking.md": (
        "The chunker splits text into overlapping token windows. "
        "Defaults are 500 tokens per chunk with 50 tokens of overlap. "
        "Cuts snap to sentence boundaries when possible."
    ),
}

VOCAB = ["docker", "compose", "postgres", "chunk", "tokens", "overlap"]


def fake_embed(text: str) -> list[float]:
    words = text.lower().split()
    return [float(sum(w.startswith(v) for w in words)) for v in VOCAB]


def run_pipeline(question: str, top_k: int = 2):
    """Ingest DOCS, embed the question, return ranked chunks."""
    candidates = []
    for filename, text in DOCS.items():
        for i, piece in enumerate(
            chunk_text(text, chunk_size=20, chunk_overlap=5,
                       use_tiktoken=False, respect_sentence_boundaries=False)
        ):
            candidates.append(
                {"id": f"{filename}#{i}", "filename": filename,
                 "content": piece, "embedding": fake_embed(piece)}
            )
    return rank_chunks(fake_embed(question), candidates, top_k=top_k)


def test_pipeline_retrieves_relevant_doc():
    ranked = run_pipeline("How do I deploy with docker compose?")
    assert ranked[0]["filename"] == "deploy.md"
    assert "docker" in ranked[0]["content"].lower()


def test_pipeline_chunking_question():
    ranked = run_pipeline("What is the chunk overlap default?")
    assert ranked[0]["filename"] == "chunking.md"
    assert "overlap" in ranked[0]["content"].lower()


def test_pipeline_top_k_and_scores():
    ranked = run_pipeline("docker postgres tokens", top_k=3)
    assert len(ranked) == 3
    scores = [r["score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)
