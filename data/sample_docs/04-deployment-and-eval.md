# Deployment and evaluation

## Running with Docker Compose

The project ships a `docker-compose.yml` with two services:

- `db` — `pgvector/pgvector:pg16`, with a healthcheck and a named volume
  so vectors survive container restarts.
- `api` — the FastAPI app, built from the `Dockerfile`. The image
  pre-downloads the embedding model at build time so the first request
  does not pay the download cost and the container can run without egress.

Start everything with `docker compose up --build`, then ingest the sample
docs and ask a question:

```bash
curl -F "file=@data/sample_docs/01-architecture.md" http://localhost:8000/ingest
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does the query path work?"}'
```

Copy `.env.example` to `.env` to tune `CHUNK_SIZE`, `CHUNK_OVERLAP`,
`TOP_K`, and the optional `ANTHROPIC_API_KEY`.

## Evaluation

The `eval/` directory holds a golden Q&A set (`golden.json`, ten
question/answer pairs grounded in the sample docs) and `eval.py`, which
ingests the sample docs, embeds every question, and reports retrieval
hit-rate@k plus the rank of the first chunk from the gold source document.

Run it with `python eval/eval.py`. It downloads the embedding model on the
first run (about 80 MB) and is otherwise offline. An optional `--db` mode
runs retrieval through Postgres/pgvector instead of the in-memory ranker,
so you can verify the SQL path returns the same ordering.

Hit-rate@k is the fraction of questions for which at least one of the top-k
retrieved chunks comes from the gold document. Mean reciprocal rank (MRR)
averages 1/rank of the first gold chunk, rewarding systems that put the
right chunk first rather than fifth. Both metrics measure the retriever
only — chunking plus embeddings — not the generator, which is why the eval
needs no Anthropic key.

## Design decisions worth defending

- pgvector over a managed vector DB: one database, transactional
  consistency, no extra service to operate at this scale.
- Token-aware chunking with tiktoken and a word-count fallback: correct
  budgets when the dependency is present, graceful degradation when not.
- Lazy model loading: tests and CLIs that never embed pay zero cost.
- Optional generation: the API is useful (retrieval + notice) with no
  API keys at all, which also makes the whole pipeline testable offline.
