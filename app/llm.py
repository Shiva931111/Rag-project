from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from app.config import settings


def get_chat_llm(temperature: float = 0.1):
    provider = settings.llm_provider.lower()
    if provider == "openai":
        return ChatOpenAI(
            model=settings.llm_model,
            temperature=temperature,
            api_key=settings.openai_api_key,
        )
    if provider == "ollama":
        return ChatOllama(
            model=settings.llm_model,
            temperature=temperature,
            base_url=settings.ollama_base_url,  # type: ignore[arg-type]
        )
    raise ValueError(f"Unsupported llm_provider: {settings.llm_provider}")


def build_query_rewriter() -> Runnable:
    prompt = ChatPromptTemplate.from_template(
        """
You rewrite user questions for retrieval quality.
Keep intent unchanged. Expand abbreviations, include bilingual hints for English/Hindi/Hinglish if useful.
Return only one rewritten query line.

User query: {query}
"""
    )
    return prompt | get_chat_llm(temperature=0.0)


def build_answer_chain() -> Runnable:
    prompt = ChatPromptTemplate.from_template(
        """
You are a careful RAG assistant.
Answer using only the provided context.
If context is insufficient, say what is missing.
When user query is Hindi/Hinglish, respond naturally in same style unless asked otherwise.

Context:
{context}

Question:
{query}
"""
    )
    return prompt | get_chat_llm(temperature=0.2)


def build_context_compressor() -> Runnable:
    prompt = ChatPromptTemplate.from_template(
        """
You are filtering context for a question.
Given one chunk, either output:
1) a concise compressed version preserving facts relevant to the question
2) DROP (if irrelevant)

Question: {query}
Chunk: {chunk}
"""
    )
    return prompt | get_chat_llm(temperature=0.0)

