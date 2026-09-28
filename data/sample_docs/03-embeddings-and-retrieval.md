# Embeddings and retrieval

## The embedding model: all-MiniLM-L6-v2

This project embeds with `all-MiniLM-L6-v2` from sentence-transformers. The
choice is a deliberate speed/size/quality tradeoff:

- Size: 80 MB, 22 million parameters. It downloads in seconds and fits in
  any container without GPU.
- Speed: thousands of sentences per second on CPU. Ingest latency stays
  low even for hundred-page documents.
- Quality: 384-dimensional vectors that are competitive on standard
  retrieval benchmarks for English prose, within a few points of models
  ten times its size.
- Output dimension 384 keeps the pgvector column narrow, which keeps the
  index small and distance computations fast.

When to switch: move to a larger model (e.g. `all-mpnet-base-v2` at 768
dimensions, or a multilingual model like `paraphrase-multilingual-mpnet`)
when retrieval quality plateaus and you have evidence — from the eval
harness — that misses are caused by the embedding rather than chunking.
Switching models requires re-embedding every stored chunk, because vectors
from different models are not comparable.

The model is lazy-loaded once per process and cached. Importing the
embeddings module never triggers a download; only the first `embed_texts`
call does.

## Retrieval: cosine similarity over pgvector

Each chunk's embedding is stored in a PostgreSQL `vector(384)` column via
the pgvector extension. At query time the question is embedded with the
same model and Postgres computes cosine distance (`<=>`) against every
chunk, returning the top-k by `ORDER BY distance LIMIT k`.

Cosine similarity measures the angle between vectors, ignoring magnitude,
which is the right geometry for text: two chunks about the same topic point
in the same direction even if one is longer. The API reports similarity
scores (1 − distance), so higher means more relevant.

Storing vectors next to the text in Postgres — instead of a separate vector
database — keeps one source of truth, one backup story, and transactional
consistency between a chunk and its embedding. A dedicated vector database
becomes worth it past roughly a million vectors, or when you need
approximate-index features Postgres does not offer.

## Top-k selection

`TOP_K` defaults to 5. Fewer chunks mean less noise for the generator but a
higher miss rate; more chunks mean better recall at the cost of context
bloat and noisier answers. Five is the standard starting point for
grounded Q&A, and the `/query` endpoint accepts a per-request override.
