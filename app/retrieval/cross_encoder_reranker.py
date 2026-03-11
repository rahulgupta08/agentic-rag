from typing import List, Dict
from sentence_transformers import CrossEncoder
from .base_reranker import BaseReranker
import asyncio


class CrossEncoderReranker(BaseReranker):

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = None
    ):
        """
        model_name: HuggingFace cross encoder model
        device: "cpu" or "cuda"
        """
        self.model = CrossEncoder(model_name, device=device)

    async def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:

        if not documents:
            return []

        # Create (query, document) pairs
        pairs = [(query, doc["text"]) for doc in documents]

        # Batch scoring
        scores = await asyncio.to_thread(
                        self.model.predict,
                        pairs
                    )

        # Attach scores
        for score, doc in zip(scores, documents):
            doc["rerank_score"] = float(score)

        # Sort by rerank_score (descending)
        documents.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return documents[:top_k]