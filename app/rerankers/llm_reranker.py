from typing import List, Dict
from app.rerankers.base_reranker import BaseReranker


class LLMReranker(BaseReranker):

    def __init__(self, llm):
        self.llm = llm

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """Synchronous rerank for RAG mode"""
        scored_docs = []

        for doc in documents:
            score = self._score_sync(query, doc["text"])
            doc["reranker_score"] = score
            scored_docs.append(doc)

        # Sort descending
        scored_docs.sort(
            key=lambda x: x["reranker_score"],
            reverse=True
        )

        return scored_docs[:top_k]

    async def arerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """Asynchronous rerank for Agent mode"""
        scored_docs = []

        for doc in documents:
            score = await self._score_async(query, doc["text"])
            doc["reranker_score"] = score
            scored_docs.append(doc)

        # Sort descending
        scored_docs.sort(
            key=lambda x: x["reranker_score"],
            reverse=True
        )

        return scored_docs[:top_k]

    def _score_sync(self, query: str, document: str) -> float:
        """Synchronous scoring for RAG mode"""
        prompt = self._build_prompt(query, document)
        response = self.llm.invoke(prompt)

        try:
            return float(response.strip())
        except:
            return 0.0

    async def _score_async(self, query: str, document: str) -> float:
        """Asynchronous scoring for Agent mode"""
        prompt = self._build_prompt(query, document)
        response = await self.llm.ainvoke(prompt)

        try:
            return float(response.strip())
        except:
            return 0.0

    def _build_prompt(self, query: str, document: str) -> str:
        """Helper method to build the scoring prompt"""
        return f"""
        You are a relevance scoring system.

        Query:
        {query}

        Document:
        {document}

        Score relevance from 0 to 1.
        Return only the number.
        """