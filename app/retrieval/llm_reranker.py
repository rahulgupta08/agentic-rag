from typing import List, Dict
from .base_reranker import BaseReranker


class LLMReranker(BaseReranker):

    def __init__(self, llm):
        self.llm = llm

    async def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:

        scored_docs = []

        for doc in documents:
            score = await self._score(query, doc["text"])
            doc["rerank_score"] = score
            scored_docs.append(doc)

        # Sort descending
        scored_docs.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return scored_docs[:top_k]

    async def _score(self, query: str, document: str) -> float:

        prompt = f"""
        You are a relevance scoring system.

        Query:
        {query}

        Document:
        {document}

        Score relevance from 0 to 1.
        Return only the number.
        """

        response = await self.llm.invoke(prompt)

        try:
            return float(response.content.strip())
        except:
            return 0.0