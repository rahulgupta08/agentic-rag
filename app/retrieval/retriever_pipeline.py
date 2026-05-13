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
        rerank_quality_threshold: float = 0.6
    ):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.query_transformer = query_transformer
        self.query_expander = query_expander
        self.post_processors = post_processors or []
        self.logger = logger or logging.getLogger(__name__)
        self.confidence_threshold = confidence_threshold
        self.rerank_quality_threshold = rerank_quality_threshold

    def retrieve(self, query: str, top_k: int = 5, skip_rerank: bool = False):
        """Synchronous retrieve for RAG mode"""
        self.logger.info(f"Original Query: {query}")

        # -------------------------
        # Step 1 — Query Rewrite
        # -------------------------
        if self.query_transformer:
            queries = self.query_transformer.transform(query)
            query = queries[0]  # rewriter returns single-item list
            self.logger.info(f"Rewritten Query: {query}")

        # -------------------------
        # Step 2 — Initial Retrieval
        # -------------------------
        # Retrieve more documents initially if we might rerank
        initial_k = top_k * 2 if (self.reranker and not skip_rerank) else top_k
        documents = self.base_retriever.retrieve(query, initial_k)
        
        self.logger.info("Retrieved %d initial documents", len(documents))

        # -------------------------
        # Step 3 — Rerank
        # -------------------------
        if self.reranker and not skip_rerank:
            # Check if documents meet quality threshold for reranking
            if self._meets_quality_threshold(documents):
                self.logger.info("Quality threshold met, performing reranking")
                documents = self.reranker.rerank(
                    query,
                    documents,
                    top_k=top_k
                )
            else:
                self.logger.info("Quality threshold not met, skipping reranking")
                documents = documents[:top_k]

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

            expanded_queries = self.query_expander.transform(query)

            self.logger.info(
                "[RETRIEVER] Expanded Queries (%d): %s",
                len(expanded_queries),
                expanded_queries
            )

            extra_docs = []
            for q in expanded_queries:
                docs = self.base_retriever.retrieve(q, top_k)
                extra_docs.extend(docs)

            self.logger.info(
                "[RETRIEVER] Retrieved extra docs=%d",
                len(extra_docs)
            )

            # Merge
            documents = self._merge_documents(documents, extra_docs)

            # Rerank again if available
            if self.reranker:
                documents = self.reranker.rerank(  # Use sync rerank
                    query,
                    documents,
                    top_k=top_k
                )

        # -------------------------
        # Step 6 — Post-Processing
        # -------------------------
        for processor in self.post_processors:
            documents = processor.process(documents)

        return documents

    async def aretrieve(self, query: str, top_k: int = 5, skip_rerank: bool = False):
        """Asynchronous retrieve for Agent mode"""
        self.logger.info(f"Original Query: {query}")

        # -------------------------
        # Step 1 — Query Rewrite
        # -------------------------
        if self.query_transformer:
            queries = await self.query_transformer.transform_async(query)
            query = queries[0]  # rewriter returns single-item list
            self.logger.info(f"Rewritten Query: {query}")

        # -------------------------
        # Step 2 — Initial Retrieval
        # -------------------------
        # Retrieve more documents initially if we might rerank
        initial_k = top_k * 2 if (self.reranker and not skip_rerank) else top_k
        documents = await self.base_retriever.aretrieve(query, initial_k)
        
        logger.info("Retrieved %d initial documents", len(documents))

        # -------------------------
        # Step 3 — Rerank
        # -------------------------
        if self.reranker and not skip_rerank:
            # Check if documents meet quality threshold for reranking
            if self._meets_quality_threshold(documents):
                logger.info("Quality threshold met, performing reranking")
                documents = await self.reranker.arerank(
                    query,
                    documents,
                    top_k=top_k
                )
            else:
                logger.info("Quality threshold not met, skipping reranking")
                documents = documents[:top_k]

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

            expanded_queries = await self.query_expander.transform_async(query)

            self.logger.info(
                "[RETRIEVER] Expanded Queries (%d): %s",
                len(expanded_queries),
                expanded_queries
            )

            extra_docs = []
            for q in expanded_queries:
                docs = await self.base_retriever.aretrieve(q, top_k)
                extra_docs.extend(docs)

            self.logger.info(
                "[RETRIEVER] Retrieved extra docs=%d",
                len(extra_docs)
            )

            # Merge
            documents = self._merge_documents(documents, extra_docs)

            # Rerank again if available
            if self.reranker:
                documents = await self.reranker.arerank(  # Use async rerank
                    query,
                    documents,
                    top_k=top_k
                )

        # -------------------------
        # Step 6 — Post-Processing
        # -------------------------
        for processor in self.post_processors:
            documents = processor.process(documents)

        return documents

    def _compute_confidence(self, documents: List[Dict]) -> float:
        """Helper method to compute retrieval confidence"""
        if not documents:
            return 0.0
        
        # Average similarity score across top documents
        scores = [doc.get("score", 0.0) for doc in documents]
        return sum(scores) / len(scores)

    def _merge_documents(self, docs1: List[Dict], docs2: List[Dict]) -> List[Dict]:
        """Helper method to merge document lists, removing duplicates"""
        seen = set()
        merged = []

        for doc in docs1 + docs2:
            doc_id = doc.get("id")
            if doc_id not in seen:
                seen.add(doc_id)
                merged.append(doc)

        return merged

    def _meets_quality_threshold(self, documents: List[Dict]) -> bool:
        """Check if documents meet quality threshold for reranking."""
        if not documents:
            return False
            
        # Calculate average similarity score
        scores = [doc.get("score", 0.0) for doc in documents]
        avg_score = sum(scores) / len(scores)
        
        self.logger.info(
            "Document quality check: avg_score=%.3f threshold=%.3f",
            avg_score,
            self.rerank_quality_threshold
        )
        
        return avg_score > self.rerank_quality_threshold