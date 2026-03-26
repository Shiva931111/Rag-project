"""
Backward-compatible wrappers around the upgraded pipeline.
"""

from app.ingestion import IngestionService
from app.retrieval import RetrievalService


def process_pdf(pdf_path: str) -> str:
    ingestion = IngestionService()
    ingestion.ingest("pdf", pdf_path, {"source": pdf_path})
    return "indexed"


def ask_question(_db: str, question: str) -> str:
    retrieval = RetrievalService()
    result = retrieval.answer(question)
    return result["answer"]