import json
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.config import settings
from app.llm import build_answer_chain, build_context_compressor, build_query_rewriter


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class RetrievalService:
    def __init__(self) -> None:
        self.embedding = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        self.vector_store = Chroma(
            collection_name=settings.collection_name,
            persist_directory=settings.chroma_dir,
            embedding_function=self.embedding,
        )
        self.rewriter = build_query_rewriter()
        self.answer_chain = build_answer_chain()
        self.compressor = build_context_compressor()
        self.reranker = CrossEncoder(settings.reranker_model)
        self.corpus_path = Path(settings.corpus_store_path)

    def _get_corpus_docs(self) -> list[Document]:
        if not self.corpus_path.exists():
            return []
        raw = json.loads(self.corpus_path.read_text(encoding="utf-8"))
        return [Document(page_content=item["text"], metadata=item.get("metadata", {})) for item in raw]

    def rewrite_query(self, query: str) -> str:
        result = self.rewriter.invoke({"query": query})
        rewritten = getattr(result, "content", str(result)).strip()
        return rewritten or query

    def _vector_search(self, query: str, metadata_filter: dict[str, Any] | None) -> list[Document]:
        return self.vector_store.similarity_search(
            query=query,
            k=settings.default_top_k,
            filter=metadata_filter or None,
        )

    def _keyword_search(self, query: str) -> list[Document]:
        docs = self._get_corpus_docs()
        if not docs:
            return []
        tokenized = [_tokenize(d.page_content) for d in docs]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(_tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: settings.default_top_k]
        return [docs[i] for i in ranked_idx if scores[i] > 0]

    def _dedupe_docs(self, docs: list[Document]) -> list[Document]:
        seen: set[str] = set()
        unique: list[Document] = []
        for d in docs:
            key = f"{d.metadata.get('source','unknown')}::{d.metadata.get('chunk_index',-1)}::{d.page_content[:80]}"
            if key not in seen:
                unique.append(d)
                seen.add(key)
        return unique

    def hybrid_retrieve(self, query: str, metadata_filter: dict[str, Any] | None = None) -> list[Document]:
        vector_docs = self._vector_search(query, metadata_filter)
        keyword_docs = self._keyword_search(query)
        return self._dedupe_docs(vector_docs + keyword_docs)

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return []
        pairs = [[query, d.page_content] for d in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: settings.rerank_top_k]]

    def compress_context(self, query: str, docs: list[Document]) -> list[Document]:
        compressed: list[Document] = []
        for doc in docs[: settings.max_context_chunks]:
            result = self.compressor.invoke({"query": query, "chunk": doc.page_content})
            text = getattr(result, "content", str(result)).strip()
            if text and text.upper() != "DROP":
                compressed.append(Document(page_content=text, metadata=doc.metadata))
        return compressed

    def answer(self, query: str, metadata_filter: dict[str, Any] | None = None) -> dict[str, Any]:
        rewritten = self.rewrite_query(query)
        retrieved = self.hybrid_retrieve(rewritten, metadata_filter=metadata_filter)
        reranked = self.rerank(rewritten, retrieved)
        compressed = self.compress_context(rewritten, reranked)
        context = "\n\n".join(
            [f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}" for d in compressed]
        )
        llm_result = self.answer_chain.invoke({"query": query, "context": context or "No context found."})
        answer = getattr(llm_result, "content", str(llm_result))
        return {
            "query": query,
            "rewritten_query": rewritten,
            "answer": answer,
            "sources": [d.metadata for d in compressed],
        }

