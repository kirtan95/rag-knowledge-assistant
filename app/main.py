"""rag-knowledge-assistant: RAG Q&A over your own documents.

Endpoints:
    POST /ingest            upload .txt/.md/.pdf -> extract, chunk, embed, store
    POST /query             ask a question -> retrieve top-k -> answer (Claude if key set)
    GET  /documents         list ingested documents
    DELETE /documents/{id}  delete a document and its chunks
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.chunking import chunk_text, count_tokens
from app.config import settings
from app.database import engine, get_session
from app.embeddings import embed_texts
from app.extraction import SUPPORTED_EXTENSIONS, UnsupportedFormatError, extract_text
from app.generation import NO_KEY_NOTICE, generate_answer
from app.models import Base, Chunk, Document
from app.retrieval import retrieve_from_db
from app.schemas import (
    DocumentOut,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The pgvector extension must exist before the vector(384) column DDL runs.
    # (The compose DB user is a superuser, so this works on first boot.
    # For production migrations use Alembic.)
    from sqlalchemy import text as sql_text

    with engine.begin() as conn:
        conn.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    yield


app = FastAPI(
    title="RAG Knowledge Assistant",
    description="Ingest your documents, ask questions, get grounded answers.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest(file: UploadFile = File(...), session: Session = Depends(get_session)):
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(data)} bytes (limit {settings.max_upload_bytes}).",
        )
    filename = file.filename or "upload"
    try:
        text = extract_text(filename, data)
    except UnsupportedFormatError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    if not text.strip():
        raise HTTPException(status_code=422, detail="No extractable text found in upload.")

    chunks = chunk_text(
        text,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    token_counts = [count_tokens(c) for c in chunks]
    embeddings = embed_texts(chunks)

    doc = Document(
        filename=filename,
        content_type=file.content_type or "text/plain",
        char_count=len(text),
        chunk_count=len(chunks),
    )
    session.add(doc)
    session.flush()  # assign doc.id before inserting chunks
    for i, (content, vec, tok) in enumerate(zip(chunks, embeddings, token_counts)):
        session.add(
            Chunk(
                document_id=doc.id,
                chunk_index=i,
                content=content,
                token_count=tok,
                embedding=vec,
            )
        )
    session.commit()

    return IngestResponse(
        document_id=doc.id,
        filename=doc.filename,
        chunks_created=len(chunks),
        tokens_total=sum(token_counts),
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, session: Session = Depends(get_session)):
    from app.embeddings import embed_text  # lazy: avoids model load at import time

    top_k = req.top_k or settings.top_k
    query_vector = embed_text(req.question)
    hits = retrieve_from_db(session, query_vector, top_k=top_k)

    chunks = [
        RetrievedChunk(
            chunk_id=h["id"],
            document_id=h["document_id"],
            filename=h["filename"],
            chunk_index=h["chunk_index"],
            content=h["content"],
            score=h["score"],
        )
        for h in hits
    ]

    if settings.anthropic_api_key:
        answer = generate_answer(req.question, hits)
        return QueryResponse(
            question=req.question, answer=answer, generation_used=True, chunks=chunks
        )
    return QueryResponse(
        question=req.question,
        answer=None,
        generation_used=False,
        notice=NO_KEY_NOTICE,
        chunks=chunks,
    )


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(session: Session = Depends(get_session)):
    docs = session.scalars(select(Document).order_by(Document.created_at.desc())).all()
    return docs


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: UUID, session: Session = Depends(get_session)):
    doc = session.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    session.delete(doc)  # chunks cascade via delete-orphan
    session.commit()
    return None
