import json
import uuid
from pathlib import Path
from typing import Any

import requests
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


class IngestionService:
    def __init__(self) -> None:
        self.embedding = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        self.vector_store = Chroma(
            collection_name=settings.collection_name,
            persist_directory=settings.chroma_dir,
            embedding_function=self.embedding,
        )
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
        self.corpus_path = Path(settings.corpus_store_path)
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.corpus_path.exists():
            self.corpus_path.write_text("[]", encoding="utf-8")

    def _load_docs_from_github(self, repo_url: str) -> list[Document]:
        # Lightweight GitHub loader that always works for public repos.
        cleaned = repo_url.replace("https://github.com/", "").strip("/")
        if len(cleaned.split("/")) < 2:
            raise ValueError("GitHub URL must look like https://github.com/org/repo")
        owner, repo = cleaned.split("/")[:2]
        readme_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md"
        response = requests.get(readme_url, timeout=20)
        if response.status_code != 200:
            readme_url = f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.md"
            response = requests.get(readme_url, timeout=20)
        response.raise_for_status()
        return [
            Document(
                page_content=response.text,
                metadata={"source": repo_url, "source_type": "github", "topic": repo},
            )
        ]

    def load_documents(self, source_type: str, source_value: str) -> list[Document]:
        source_type = source_type.lower()
        if source_type == "pdf":
            docs = PyPDFLoader(source_value).load()
        elif source_type == "website":
            docs = WebBaseLoader(source_value).load()
        elif source_type == "github":
            docs = self._load_docs_from_github(source_value)
        else:
            raise ValueError("source_type must be one of: pdf, website, github")
        return docs

    def _save_to_corpus(self, documents: list[Document]) -> None:
        existing = json.loads(self.corpus_path.read_text(encoding="utf-8"))
        for doc in documents:
            existing.append({"text": doc.page_content, "metadata": doc.metadata})
        self.corpus_path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def ingest(self, source_type: str, source_value: str, extra_metadata: dict[str, Any] | None = None) -> int:
        documents = self.load_documents(source_type, source_value)
        chunks = self.splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata = {
                **chunk.metadata,
                **(extra_metadata or {}),
                "source_type": source_type,
                "chunk_index": i,
            }
        ids = [str(uuid.uuid4()) for _ in chunks]
        self.vector_store.add_documents(chunks, ids=ids)
        self._save_to_corpus(chunks)
        return len(chunks)

