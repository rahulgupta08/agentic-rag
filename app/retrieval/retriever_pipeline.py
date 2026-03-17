
from app.bootstrap import bootstrap
bootstrap()
import logging
logger = logging.getLogger(__name__)
from app.retrieval.base_retriever import BaseRetriever
from typing import List, Dict


class RetrieverPipeline(BaseRetriever):

    def __init__(
        self,
        base_retriever,
        reranker=None,
        query_transformer=None,
        query_expander=None,   
        post_processors=None,
        logger=None,
        confidence_threshold: float = 0.5,
    ):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.query_transformer = query_transformer
        self.query_expander = query_expander   
        self.post_processors = post_processors or []
        self.logger = logger or logging.getLogger(__name__)
        self.confidence_threshold = confidence_threshold

    async def retrieve(self, query: str, top_k: int = 5):

        self.logger.info(f"Original Query: {query}")

        # -------------------------
        # Step 1 — Query Rewrite
        # -------------------------
        if self.query_transformer:
            queries = await self.query_transformer.transform(query)
            query = queries[0]  # rewriter returns single-item list
            self.logger.info(f"Rewritten Query: {query}")

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
        # Step 4 — Confidence
        # -------------------------
        confidence = self._compute_confidence(documents)

        self.logger.info(
            "Retrieval confidence: %.4f | threshold: %.4f",
            confidence,
            self.confidence_threshold
        )

        # -------------------------
        # Step 5 — Adaptive Retrieval
        # -------------------------
        if confidence < self.confidence_threshold and self.query_expander:

            self.logger.warning(
                        "[RETRIEVER] Low confidence=%.4f → triggering expansion",
                        confidence
                    )

            expanded_queries = await self.query_expander.transform(query)

            self.logger.info(
                "[RETRIEVER] Expanded Queries (%d): %s",
                len(expanded_queries),
                expanded_queries
            )

            extra_docs = []

            for q in expanded_queries:
                docs = await self.base_retriever.retrieve(q, top_k)
                extra_docs.extend(docs)

            self.logger.info(
                "[RETRIEVER] Retrieved extra docs=%d",
                len(extra_docs)
            )

            # Merge
            documents = self._merge_documents(documents, extra_docs)

            self.logger.info(
                "[RETRIEVER] After merge docs=%d",
                len(documents)
            )

            # Final rerank
            if self.reranker:
                documents = await self.reranker.rerank(
                    query,
                    documents,
                    top_k=top_k
                )

            self.logger.info(
                "[RETRIEVER] Final docs after rerank=%d",
                len(documents)
            )

        # -------------------------
        # Step 6 — Post-processing
        # -------------------------
        for processor in self.post_processors:
            documents = processor.process(documents)

        return documents

    # -------------------------
    # Helpers
    # -------------------------

    def _compute_confidence(self, documents: List[Dict]) -> float:

        if not documents:
            return 0.0

        weighted_scores = []
        weights = []

        for i, doc in enumerate(documents):

            if not isinstance(doc, dict):
                continue

            score = None

            # -------------------------
            # Normalize scores
            # -------------------------

            if "reranker_score" in doc:
                score = self._normalize_reranker_score(doc["reranker_score"])

            elif "score" in doc:
                score = float(doc["score"])  # already 0–1

            if score is None:
                continue

            # -------------------------
            # Top-weighted importance
            # -------------------------
            weight = 1 / (i + 1)

            weighted_scores.append(score * weight)
            weights.append(weight)

        if not weights:
            return 0.0

        return sum(weighted_scores) / sum(weights)

    def _merge_documents(
        self,
        docs1: List[Dict],
        docs2: List[Dict]
    ) -> List[Dict]:

        seen = set()
        merged = []

        for doc in docs1 + docs2:

            text = doc.get("text", "").strip()

            if not text:
                continue

            if text in seen:
                continue

            seen.add(text)
            merged.append(doc)

        return merged
    
    def _normalize_reranker_score(self, score: float) -> float:
        """
        Normalize reranker scores to 0–1.
        Handles:
        - CrossEncoder (-10 to +10)
        - LLM (0 to 1)
        """

        try:
            score = float(score)
        except:
            return 0.0

        # If already in [0,1]
        if 0.0 <= score <= 1.0:
            return score

        # CrossEncoder normalization (sigmoid)
        import math
        return 1 / (1 + math.exp(-score))