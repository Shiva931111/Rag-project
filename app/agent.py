from app.llm import get_chat_llm
from app.retrieval import RetrievalService


class RagAgent:
    def __init__(self, retrieval_service: RetrievalService) -> None:
        self.retrieval = retrieval_service

    def run(self, query: str) -> str:
        router_prompt = (
            "Decide if this question needs retrieval over indexed sources.\n"
            "Return exactly one word: RAG or GENERAL.\n\n"
            f"Question: {query}"
        )
        decision = get_chat_llm(temperature=0.0).invoke(router_prompt).content.strip().upper()
        if "RAG" in decision:
            return self.retrieval.answer(query)["answer"]
        return get_chat_llm(temperature=0.3).invoke(query).content

