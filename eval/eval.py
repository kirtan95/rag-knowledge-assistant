"""Retrieval eval: hit-rate@k and mean reciprocal rank over a golden Q&A set.

What it does:
  1. Loads the markdown docs from data/sample_docs/.
  2. Chunks them with the real chunker (app/chunking.py).
  3. Embeds every chunk with the real embedding model
     (downloads all-MiniLM-L6-v2, ~80MB, on first run; cached afterwards).
  4. For each question in eval/golden.json, embeds the question, ranks all
     chunks with the pure-Python cosine ranker, and records the rank of the
     first chunk from the gold source document.
  5. Reports hit-rate@1/@3/@5 and MRR.

Modes:
  --mode memory (default): rank in memory. Offline except the model download.
  --mode db:             also write chunks to Postgres/pgvector and verify the
                         SQL ranking matches the in-memory ranking. Needs
                         DATABASE_URL pointing at a running pgvector instance
                         (e.g. `docker compose up db`).

The eval measures the retriever only (chunking + embeddings). No Anthropic
key is needed.

Usage:
    python eval/eval.py [--mode memory|db] [--top-k 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.chunking import chunk_text, count_tokens  # noqa: E402
from app.config import settings  # noqa: E402
from app.embeddings import embed_texts  # noqa: E402
from app.retrieval import rank_chunks  # noqa: E402


def load_corpus(docs_dir: Path) -> list[dict]:
    """Chunk every .md doc; each chunk remembers its source filename."""
    chunks = []
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for i, piece in enumerate(
            chunk_text(text, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        ):
            chunks.append(
                {
                    "id": f"{path.name}#{i}",
                    "filename": path.name,
                    "chunk_index": i,
                    "content": piece,
                    "tokens": count_tokens(piece),
                }
            )
    return chunks


def evaluate(golden: list[dict], chunks: list[dict], top_k: int) -> dict:
    questions = [g["question"] for g in golden]
    q_vectors = embed_texts(questions)
    c_vectors = embed_texts([c["content"] for c in chunks])
    for chunk, vec in zip(chunks, c_vectors):
        chunk["embedding"] = vec

    per_question = []
    for g, q_vec in zip(golden, q_vectors):
        ranked = rank_chunks(q_vec, chunks, top_k=top_k)
        gold_rank = next(
            (i + 1 for i, c in enumerate(ranked) if c["filename"] == g["source_doc"]),
            None,
        )
        per_question.append(
            {"id": g["id"], "question": g["question"], "gold_rank": gold_rank}
        )

    def hit_rate(k: int) -> float:
        hits = sum(1 for q in per_question if q["gold_rank"] is not None and q["gold_rank"] <= k)
        return hits / len(per_question)

    mrr = sum(1.0 / q["gold_rank"] for q in per_question if q["gold_rank"]) / len(per_question)
    return {
        "n_questions": len(per_question),
        "n_chunks": len(chunks),
        "hit_rate@1": hit_rate(1),
        "hit_rate@3": hit_rate(3),
        "hit_rate@5": hit_rate(5),
        "mrr": mrr,
        "per_question": per_question,
    }


def verify_db_ranking(chunks: list[dict], golden: list[dict], top_k: int) -> None:
    """Write chunks to pgvector and check the SQL ranking agrees with memory."""
    from sqlalchemy import text as sql_text

    from app.database import SessionLocal
    from app.models import Base, Chunk, Document
    from app.retrieval import retrieve_from_db
    from app.embeddings import embed_text

    session = SessionLocal()
    try:
        session.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.drop_all(session.bind)
        Base.metadata.create_all(session.bind)
        doc = Document(filename="eval-corpus", content_type="text/markdown",
                       char_count=0, chunk_count=len(chunks))
        session.add(doc)
        session.flush()
        for c in chunks:
            session.add(Chunk(document_id=doc.id, chunk_index=c["chunk_index"],
                              content=c["content"], token_count=c["tokens"],
                              embedding=c["embedding"]))
        session.commit()

        mismatches = 0
        for g in golden:
            q_vec = embed_text(g["question"])
            mem = [c["id"] for c in rank_chunks(q_vec, chunks, top_k=top_k)]
            db = [
                f"{r['filename']}#{r['chunk_index']}"
                for r in retrieve_from_db(session, q_vec, top_k=top_k)
            ]
            if mem != db:
                mismatches += 1
                print(f"  MISMATCH {g['id']}: memory={mem} db={db}")
        print(f"db-vs-memory agreement: {len(golden) - mismatches}/{len(golden)} questions")
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG retrieval eval")
    parser.add_argument("--mode", choices=["memory", "db"], default="memory")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    docs_dir = ROOT / "data" / "sample_docs"
    golden = json.loads((ROOT / "eval" / "golden.json").read_text(encoding="utf-8"))
    print(f"Chunking {docs_dir} ...")
    chunks = load_corpus(docs_dir)
    print(f"  {len(chunks)} chunks (tokenizer: ", end="")
    from app.chunking import tokenization_mode
    print(f"{tokenization_mode()})")
    print("Embedding corpus + questions (first run downloads the model, ~80MB) ...")
    report = evaluate(golden, chunks, top_k=args.top_k)

    print("\n==== retrieval eval ====")
    print(f"questions : {report['n_questions']}")
    print(f"chunks    : {report['n_chunks']}")
    print(f"hit-rate@1: {report['hit_rate@1']:.2f}")
    print(f"hit-rate@3: {report['hit_rate@3']:.2f}")
    print(f"hit-rate@5: {report['hit_rate@5']:.2f}")
    print(f"MRR       : {report['mrr']:.3f}")
    print("\nrank of first gold chunk per question:")
    for q in report["per_question"]:
        rank = q["gold_rank"] if q["gold_rank"] is not None else "MISS"
        print(f"  {q['id']}: rank={rank}  {q['question'][:60]}")

    if args.mode == "db":
        print("\nVerifying pgvector SQL ranking matches in-memory ranking ...")
        verify_db_ranking(chunks, golden, args.top_k)


if __name__ == "__main__":
    main()
