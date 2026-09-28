"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document_id: UUID
    filename: str
    chunks_created: int
    tokens_total: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)


class RetrievedChunk(BaseModel):
    chunk_id: UUID
    document_id: UUID
    filename: str
    chunk_index: int
    content: str
    score: float  # cosine similarity, higher is better


class QueryResponse(BaseModel):
    question: str
    answer: str | None = None  # set when Claude generation ran
    generation_used: bool
    notice: str | None = None  # explains why generation was skipped
    chunks: list[RetrievedChunk]


class DocumentOut(BaseModel):
    id: UUID
    filename: str
    content_type: str
    char_count: int
    chunk_count: int
    created_at: datetime

    model_config = {"from_attributes": True}
