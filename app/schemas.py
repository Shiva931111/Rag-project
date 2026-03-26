from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2)
    use_agent: bool = True
    metadata_filter: dict[str, Any] | None = None


class QueryResponse(BaseModel):
    query: str
    rewritten_query: str
    answer: str
    sources: list[dict[str, Any]]


class UploadResponse(BaseModel):
    status: str
    message: str
    chunks_indexed: int


class HealthResponse(BaseModel):
    status: str = "ok"

