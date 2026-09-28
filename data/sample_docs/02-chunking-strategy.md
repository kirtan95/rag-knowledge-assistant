# Chunking strategy

Chunking is the least glamorous and most consequential part of a RAG
pipeline. A chunk is the unit of retrieval: whatever the retriever returns
is what the generator gets to read. Get chunking wrong and even a perfect
embedding model returns sliced-up, context-free fragments.

## Why tokens, not characters

Chunk budgets exist because embedding models and language models think in
tokens. A 500-character window of dense technical prose can be 700 tokens in
one language and 300 in another. Token budgets are stable across content;
character budgets are not. This project therefore counts with tiktoken's
`cl100k_base` encoding when tiktoken is installed, and falls back to
whitespace word counts when it is not, so the pipeline keeps working in
minimal or offline environments.

## Defaults: 500 tokens with 50 overlap

The defaults are `CHUNK_SIZE=500` and `CHUNK_OVERLAP=50`, both configurable
via environment variables. The reasoning:

- 500 tokens is roughly 350–400 words, about a page of prose. It is large
  enough to hold a complete thought — a paragraph plus its surrounding
  context — and small enough that the embedding still represents one idea
  rather than an average of many.
- 50 tokens of overlap (10%) means a fact that straddles a chunk boundary
  still appears whole in at least one chunk. Overlap is cheap insurance:
  it costs a little extra storage and embedding compute, and it measurably
  reduces boundary misses.

## Sentence-boundary snapping

A naive sliding window can cut mid-sentence, which hurts both embedding
quality and the readability of retrieved context. The chunker therefore
snaps each cut to the last sentence-ending token found in the final 40% of
the window. The search is deliberately restricted to the tail: snapping too
early would produce many undersized chunks and defeat the size budget. The
heuristic ignores lone punctuation tokens so abbreviations like "e.g." do
not trigger false boundaries. It can be disabled per call with
`respect_sentence_boundaries=False`.

## When to change the defaults

- Smaller chunks (200–300 tokens) for factoid Q&A over dense reference
  material, where answers live in single sentences.
- Larger chunks (800–1000 tokens) for narrative or legal documents where
  meaning depends on long-range context.
- More overlap (15–20%) when documents have long sentences or tables that
  frequently cross boundaries.
