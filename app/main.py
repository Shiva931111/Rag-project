import logging
import tempfile
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.agent import RagAgent
from app.config import settings
from app.ingestion import IngestionService
from app.schemas import HealthResponse, QueryRequest, QueryResponse, UploadResponse
from app.retrieval import RetrievalService


logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag-api")

app = FastAPI(title=settings.app_name)


@lru_cache(maxsize=1)
def get_ingestion_service() -> IngestionService:
    return IngestionService()


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    return RetrievalService()


@lru_cache(maxsize=1)
def get_rag_agent() -> RagAgent:
    return RagAgent(get_retrieval_service())


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile | None = File(default=None),
    source_type: str = Form(default="pdf"),
    url: str | None = Form(default=None),
    topic: str | None = Form(default=None),
) -> UploadResponse:
    try:
        source_type = source_type.lower()
        metadata: dict[str, Any] = {"topic": topic} if topic else {}

        if source_type == "pdf":
            if file is None:
                raise HTTPException(status_code=400, detail="PDF upload requires file.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                content = await file.read()
                temp_pdf.write(content)
                pdf_path = temp_pdf.name
            metadata["source"] = file.filename or "uploaded_pdf"
            count = get_ingestion_service().ingest("pdf", pdf_path, metadata)
        elif source_type in {"website", "github"}:
            if not url:
                raise HTTPException(status_code=400, detail="URL is required for website/github.")
            metadata["source"] = url
            count = get_ingestion_service().ingest(source_type, url, metadata)
        else:
            raise HTTPException(status_code=400, detail="source_type must be pdf, website, or github.")

        return UploadResponse(
            status="success",
            message=f"{source_type} ingested successfully",
            chunks_indexed=count,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        if request.use_agent:
            answer = get_rag_agent().run(request.query)
            rag_result = get_retrieval_service().answer(
                request.query, metadata_filter=request.metadata_filter
            )
            rag_result["answer"] = answer
            return QueryResponse(**rag_result)
        rag_result = get_retrieval_service().answer(
            request.query, metadata_filter=request.metadata_filter
        )
        return QueryResponse(**rag_result)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

