from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


class Settings(BaseSettings):
    app_name: str = "Production RAG API"
    app_env: str = "dev"
    log_level: str = "INFO"

    llm_provider: str = "ollama"
    llm_model: str = "mistral"
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    embedding_model: str = "intfloat/multilingual-e5-large"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chroma_dir: str = "./data/chroma"
    corpus_store_path: str = "./data/corpus/chunks.json"
    collection_name: str = "rag_chunks"

    default_top_k: int = 12
    rerank_top_k: int = 6
    max_context_chunks: int = 4

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"ollama", "openai"}:
            raise ValueError("LLM_PROVIDER must be one of: ollama, openai")
        return normalized


settings = Settings()

