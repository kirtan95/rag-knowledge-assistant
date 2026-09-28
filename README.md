# RAG Knowledge Assistant

A production-style **Retrieval-Augmented Generation (RAG)** Q&A system over your own
documents. Upload `.txt` / `.md` / `.pdf` files, ask questions in plain English, and get
answers grounded in your content — with citations.

**Why this exists.** Most RAG demos are notebooks with a FAISS index and a hardcoded
API key. This is the boring, operable version: a FastAPI service, Postgres + pgvector
as the single store for text *and* vectors, token-aware chunking, lazy model loading,
an offline-capable eval harness, and Docker Compose to run it all. It is small enough
to read in an afternoon and real enough to defend in an interview.

## Architecture

```
┌────────────────────────────── INGEST PATH ──────────────────────────────────┐
│                                                                             │
│  POST /ingest (multipart file)                                              │
│        │                                                                    │
│        ▼                                                                    │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐    ┌──────────┐  │
│  │ extraction  │───▶│   chunking   │───▶│    embeddings    │───▶│ Postgres │  │
│  │ .txt/.md/   │    │ token-aware  │    │ all-MiniLM-L6-v2 │    │ +pgvector│  │
│  │ .pdf→text   │    │ sliding win. │    │ 384-dim vectors  │    │ documents│  │
│  └─────────────┘    │ 500/50 toks  │    │ lazy-loaded      │    │ chunks   │  │
│                     └──────────────┘    └──────────────────┘    └──────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────── QUERY PATH ───────────────────────────────────┐
│                                                                             │
│  POST /query {"question": "...", "top_k": 5}                                │
│        │                                                                    │
│        ▼                                                                    │
│  ┌──────────────────┐    ┌──────────────────────────────────────┐            │
│  │    embeddings    │───▶│  Postgres: ORDER BY embedding <=> $1 │            │
│  │ embed(question)  │    │  LIMIT k   (cosine distance in SQL)  │            │
│  └──────────────────┘    └──────────────┬───────────────────────┘            │
│                                         │ top-k chunks + scores              │
│                                         ▼                                    │
│                              ┌──────────────────────┐                        │
│                              │ ANTHROPIC_API_KEY?   │                        │
│                              │  yes → Claude writes │                        │
│                              │  a cited answer      │                        │
│                              │  no  → return chunks │                        │
│                              │  + notice            │                        │
│                              └──────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quickstart

```bash
cp .env.example .env          # tune CHUNK_SIZE / CHUNK_OVERLAP / TOP_K / keys
docker compose up --build
```

Ingest the sample docs and ask a question:

```bash
curl -F "file=@data/sample_docs/01-architecture.md" http://localhost:8000/ingest

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does the query path work?"}'
```

Without `ANTHROPIC_API_KEY`, `/query` returns the retrieved chunks plus a notice that
generation needs the key — retrieval is the valuable half of RAG and works with no LLM
at all. Set the key and you get a synthesized, cited answer instead.

Other endpoints: `GET /documents`, `DELETE /documents/{id}`, `GET /health`.
Interactive docs at `http://localhost:8000/docs`.

## Project layout

```
app/
  main.py         FastAPI routes (/ingest, /query, /documents)
  chunking.py     token-aware sliding-window chunker
  embeddings.py   lazy-loaded, cached sentence-transformer
  extraction.py   .txt/.md/.pdf → plain text
  retrieval.py    cosine top-k (pure Python + pgvector SQL)
  generation.py   optional Claude answer synthesis
  models.py       SQLAlchemy 2.0 models (documents, chunks)
  schemas.py      Pydantic request/response schemas
  config.py       env-driven settings
  database.py     engine + session wiring
eval/
  golden.json     10 grounded Q/A pairs over data/sample_docs/
  eval.py         hit-rate@k + MRR harness (memory or pgvector mode)
data/sample_docs/ 4 short design docs = the eval corpus
tests/            pytest suite — offline, no DB, no model download
```

## Design deep-dives

### Chunking: tokens, windows, overlap, sentence snapping

- **Tokens, not characters.** Chunk budgets exist because embedding models and LLMs
  think in tokens. A 500-character window can be 700 tokens of dense prose or 300 of
  something else; token budgets are stable. We count with tiktoken `cl100k_base` when
  installed, and fall back to whitespace word counts when it isn't — the pipeline keeps
  working in minimal/offline environments instead of crashing on a missing dependency.
- **Defaults: 500 tokens, 50 overlap** (`CHUNK_SIZE` / `CHUNK_OVERLAP`). 500 tokens is
  ~350–400 words: big enough to hold a complete thought, small enough that the
  embedding represents one idea rather than an average of many. 50 tokens (10%) of
  overlap means a fact straddling a boundary still appears whole in at least one
  chunk — cheap insurance against boundary misses.
- **Sentence-boundary snapping.** Each cut snaps to the last sentence-ending token in
  the final 40% of the window, so chunks end on sentence boundaries instead of
  mid-word. Restricted to the tail so we don't produce undersized chunks; ignores
  lone punctuation to avoid abbreviations like "e.g."; disable with
  `respect_sentence_boundaries=False`.
- **When to change the defaults:** 200–300 tokens for factoid Q&A over dense
  reference text; 800–1000 for narrative/legal docs where meaning needs long context;
  15–20% overlap when long sentences or tables frequently cross boundaries.

### Embeddings: why `all-MiniLM-L6-v2`

A deliberate speed/size/quality tradeoff: **80 MB / 22M params** (downloads in
seconds), thousands of sentences/sec on CPU, and 384-dim vectors competitive on
retrieval benchmarks within a few points of models 10× its size. The narrow 384-dim
output keeps the pgvector column and index small and distance math fast.

**When to switch:** move to `all-mpnet-base-v2` (768-dim) or a multilingual model
only when the eval shows misses caused by the embedding rather than chunking —
and budget a full re-embed, because vectors from different models live in different
spaces and are not comparable. The model name is env-configurable (`EMBEDDING_MODEL`);
the DB column width (`vector(384)`) must match.

The model is **lazy-loaded once per process and cached** (`app/embeddings.py`):
importing the module never downloads anything; only the first `embed_texts` call
does. Tests, CLIs, and `--help` pay zero cost. The Dockerfile pre-downloads the
model at build time so containers don't pay it at request time.

### Retrieval: cosine top-k in pgvector

Chunks and their embeddings live **side by side in Postgres** — one `documents` row,
many `chunks` rows, each with a `vector(384)` column. At query time the question is
embedded with the *same* model (required: different models = different vector
spaces) and Postgres ranks with the `<=>` cosine-distance operator,
`ORDER BY distance LIMIT k`. Ranking happens inside the DB — the API never pulls the
embedding table into Python.

**Why cosine:** it measures angle, ignoring magnitude — two chunks about the same
topic point the same direction even if one is longer. The API reports similarity
(`1 − distance`), higher = more relevant.

**Top-k:** `TOP_K` defaults to 5 (overridable per request). Fewer chunks = less
noise but higher miss rate; more = better recall at the cost of context bloat. Five
is the standard starting point for grounded Q&A.

## Evaluation

`eval/golden.json` holds **10 question/answer pairs grounded in `data/sample_docs/`**
— short design docs about this very project, so the corpus is realistic and the
gold answers are checkable. Each entry records its `source_doc`.

`eval/eval.py` ingests the docs with the real chunker, embeds everything with the
real model, and reports:

- **hit-rate@k** — fraction of questions with ≥1 top-k chunk from the gold doc
- **MRR** (mean reciprocal rank) — averages `1/rank` of the first gold chunk,
  rewarding "right chunk first" over "right chunk fifth"
- per-question rank of the first gold chunk

Both metrics measure the **retriever only** (chunking + embeddings), not the
generator — no Anthropic key needed.

```bash
# in-memory ranking (default). Downloads the ~80MB model on first run, then offline.
python eval/eval.py

# also write chunks to pgvector and verify the SQL ranking matches in-memory ranking
# (needs DATABASE_URL -> a running pgvector, e.g. `docker compose up db`)
python eval/eval.py --mode db
```

## Design decisions (the interview answers)

- **pgvector over a managed vector DB.** One database, one backup story,
  transactional consistency between chunk text and its embedding, no extra service
  to operate. A dedicated vector DB earns its place past ~1M vectors or when you
  need ANN index features Postgres lacks.
- **Optional generation.** The API is useful with zero API keys — retrieval plus a
  clear notice — which also makes the entire pipeline testable offline.
- **SQLAlchemy 2.0 style** (`mapped_column`, `DeclarativeBase`) throughout; chunk
  deletion cascades from documents via `delete-orphan`.
- **SQLite is deliberately not supported**: pgvector *is* the point of this
  project, so the unit tests are written to run without any DB instead.

## Running tests

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -v
```

The suite covers the chunker (overlap exactness, boundary behavior, sentence
snapping, tiktoken + fallback paths), cosine math and ranking (order, truncation,
ties, edge cases), extraction, and an ingest→query pipeline test with fake vectors.
No test downloads the embedding model or touches a database.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg://rag:ragpassword@db:5432/ragdb` | Postgres/pgvector connection |
| `ANTHROPIC_API_KEY` | *(unset)* | Enables Claude answer generation; optional |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5` | Claude model id for generation |
| `CHUNK_SIZE` | `500` | Target tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap tokens between chunks |
| `TOP_K` | `5` | Default chunks retrieved per query |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |

## License

MIT — see [LICENSE](LICENSE).
