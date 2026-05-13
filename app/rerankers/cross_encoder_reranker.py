from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

from sentence_transformers import CrossEncoder

from app.rerankers.base_reranker import BaseReranker


logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """
    Cross-Encoder based reranker.

    Features:
    - Sync and async support
    - Batch scoring
    - Structured logging
    - Metadata score injection
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 16,
        max_workers: int = 4,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

        logger.info(
            "Initializing CrossEncoderReranker | model=%s batch_size=%d",
            model_name,
            batch_size,
        )

        self.model = CrossEncoder(model_name)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def rerank(
        self,
        query: str,
        documents: List,
        top_k: int = 5,
    ) -> List:
        """Synchronous rerank for RAG mode"""
        if not documents:
            logger.warning("Reranker received empty document list")
            return []

        logger.info(
            "Reranking started | query_length=%d docs_received=%d top_k=%d",
            len(query),
            len(documents),
            top_k,
        )

        try:
            # Step 1 — Prepare pairs
            pairs = [
                (query, self._get_doc_text(doc))
                for doc in documents
            ]

            # Step 2 — Run model directly (sync)
            scores = self._predict(pairs)

            # Step 3-5 — Process results
            return self._process_results(documents, scores, top_k)

        except Exception as e:
            logger.exception("Reranking failed")
            raise e

    async def arerank(
        self,
        query: str,
        documents: List,
        top_k: int = 5,
    ) -> List:
        """Asynchronous rerank for Agent mode"""
        if not documents:
            logger.warning("Reranker received empty document list")
            return []

        logger.info(
            "Reranking started | query_length=%d docs_received=%d top_k=%d",
            len(query),
            len(documents),
            top_k,
        )

        try:
            # Step 1 — Prepare pairs
            pairs = [
                (query, self._get_doc_text(doc))
                for doc in documents
            ]

            # Step 2 — Run model in threadpool
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                self.executor,
                self._predict,
                pairs,
            )

            # Step 3-5 — Process results
            return self._process_results(documents, scores, top_k)

        except Exception as e:
            logger.exception("Reranking failed")
            raise e

    def _predict(self, pairs: List) -> List[float]:
        """Run prediction in batches"""
        all_scores = []
        
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            scores = self.model.predict(batch)
            
            if len(batch) == 1:
                scores = [scores]
                
            all_scores.extend(scores)
            
        return all_scores

    def _get_doc_text(self, doc) -> str:
        """Extract text from document"""
        if isinstance(doc, dict):
            return doc.get("text", "")
        return getattr(doc, "page_content", "")

    def _process_results(self, documents: List, scores: List[float], top_k: int) -> List:
        """Process and sort results"""
        scored_docs = []

        for doc, score in zip(documents, scores):
            score = float(score)

            if isinstance(doc, dict):
                doc["reranker_score"] = score
            else:
                if not hasattr(doc, "metadata") or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["reranker_score"] = score

            scored_docs.append((doc, score))

        # Sort and select top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]