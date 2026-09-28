# Architecture

The RAG Knowledge Assistant is a small production-style service with two
paths through it: an ingest path and a query path. Both share the same
chunking and embedding code so that documents and questions are represented
in the same vector space.

## Ingest path

1. The client POSTs a file to `/ingest`. Supported types are `.txt`, `.md`
   and `.pdf`.
2. The API extracts plain text from the upload. PDFs are handled with pypdf;
   anything else is decoded as UTF-8.
3. The text is split into overlapping token windows by the chunker. The
   window size and overlap come from the `CHUNK_SIZE` and `CHUNK_OVERLAP`
   environment variables, defaulting to 500 and 50 tokens.
4. Each chunk is embedded with the `all-MiniLM-L6-v2` sentence-transformer,
   producing a 384-dimensional float vector.
5. One row per document goes into the `documents` table and one row per
   chunk into the `chunks` table in PostgreSQL. The chunk row stores the
   text, its token count, and its embedding in a pgvector `vector(384)`
   column, side by side.

## Query path

1. The client POSTs a question to `/query` with an optional `top_k`.
2. The question is embedded with the same model used at ingest time. Using
   the same model for both sides is required: embeddings from different
   models live in different vector spaces and cannot be compared.
3. Postgres ranks chunks by cosine distance using pgvector's `<=>`
   operator with `ORDER BY distance LIMIT k`. Ranking happens inside the
   database, so the API never pulls the full embedding table into memory.
4. If `ANTHROPIC_API_KEY` is set, the top-k chunks are passed to Claude as
   grounded context and the model writes a cited answer. If the key is not
   set, the API returns the retrieved chunks unchanged with a notice that
   generation needs the key. Retrieval is the valuable half of RAG and it
   works without any LLM.

## Components

- `app/main.py` — FastAPI routes and request wiring.
- `app/chunking.py` — token-aware sliding-window chunker.
- `app/embeddings.py` — lazy-loaded sentence-transformer service.
- `app/extraction.py` — text extraction for txt, md, and pdf uploads.
- `app/retrieval.py` — cosine top-k ranking (pure Python and pgvector SQL).
- `app/generation.py` — optional Claude answer synthesis.
- `app/models.py` — SQLAlchemy 2.0 models for documents and chunks.
