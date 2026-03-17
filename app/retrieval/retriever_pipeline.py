from app.retrieval.base_retriever import BaseRetriever
from app.bootstrap import bootstrap
bootstrap()

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class RetrieverPipeline(BaseRetriever):

    def __init__(
        self,
        base_retriever,
        reranker=None,
        query_transformer=None,
        post_processors=None,
        logger=None,
        confidence_threshold: float = 0.5,
    ):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.query_transformer = query_transformer
        self.post_processors = post_processors or []
        self.logger = logger or logging.getLogger(__name__)
        self.confidence_threshold = confidence_threshold

    async def retrieve(self, query: str, top_k: int = 5):

        self.logger.info(f"Original Query: {query}")

        # -------------------------
        # Step 1 — Query transform
        # -------------------------
        if self.query_transformer:
            query = await self.query_transformer.transform(query)
            self.logger.info(f"Transformed Query: {query}")

        # -------------------------
        # Step 2 — Initial Retrieval
        # -------------------------
        documents = await self.base_retriever.retrieve(query, top_k)

        # -------------------------
        # Step 3 — Rerank
        # -------------------------
        if self.reranker:
            documents = await self.reranker.rerank(
                query,
                documents,
                top_k=top_k
            )

        # -------------------------
        # Step 4 — Confidence Scoring
        # -------------------------
        confidence = self._compute_confidence(documents)

        self.logger.info(
            "Retrieval confidence: %.4f | threshold: %.4f",
            confidence,
            self.confidence_threshold
        )

        # -------------------------
        # Step 5 — Decision Layer
        # -------------------------
        if confidence < self.confidence_threshold:
            self.logger.warning(
                "Low confidence detected → adaptive retrieval needed (Phase 3 next step)"
            )
            # For now: return current docs (no expansion yet)
            # Next step will plug adaptive retrieval here

        # -------------------------
        # Step 6 — Post-processing
        # -------------------------
        for processor in self.post_processors:
            documents = processor.process(documents)

        return documents

    # -------------------------
    # Internal Helpers
    # -------------------------

    def _compute_confidence(self, documents: List[Dict]) -> float:
        """
        Compute confidence based on available scores.
        Priority:
        1. reranker_score
        2. rerank_score
        3. base score
        """

        if not documents:
            return 0.0

        scores = []

        for doc in documents:

            if not isinstance(doc, dict):
                continue

            score = None

            # Priority 1 — CrossEncoder
            if "reranker_score" in doc:
                score = doc["reranker_score"]

            # Fallback — base retriever
            elif "score" in doc:
                score = doc["score"]

            if score is not None:
                try:
                    scores.append(float(score))
                except:
                    continue

        if not scores:
            return 0.0

        # Simple average (stable + sufficient for now)
        return sum(scores) / len(scores)