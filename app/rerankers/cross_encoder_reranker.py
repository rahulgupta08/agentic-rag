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
    - Async-compatible (threadpool execution)
    - Batch scoring
    - Structured logging (no print statements)
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

    async def rerank(
        self,
        query: str,
        documents: List,
        top_k: int = 5,
    ) -> List:

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

            # Step 3 — Attach scores
            scored_docs = []

            for doc, score in zip(documents, scores):

                score = float(score)

                #  Handle dict documents
                if isinstance(doc, dict):
                    doc["reranker_score"] = score

                #  Handle object documents
                else:
                    if not hasattr(doc, "metadata") or doc.metadata is None:
                        doc.metadata = {}

                    doc.metadata["reranker_score"] = score

                scored_docs.append((doc, score))

            # Step 4 — Sort
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # Step 5 — Select top_k
            reranked_docs = [doc for doc, _ in scored_docs[:top_k]]

            for doc in reranked_docs:

                score = (
                    doc.get("reranker_score")
                    if isinstance(doc, dict)
                    else doc.metadata.get("reranker_score")
                )

                text = (
                    doc.get("text")
                    if isinstance(doc, dict)
                    else getattr(doc, "page_content", "")
                )

                logger.info(
                    "ReRanked Docs Scores: %s | %s",
                    score,
                    text[:60],
                )

   
            logger.debug(
                "Reranking completed | returned_docs=%d",
                len(reranked_docs),
            )

            return reranked_docs

        except Exception as e:
            logger.exception("Reranking failed | error=%s", str(e))
            
            raise

    # -------------------------
    # Internal Helpers
    # -------------------------

    def _predict(self, pairs: List):
        """
        Blocking model inference.
        Runs inside threadpool.
        """
        return self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

    def _get_doc_text(self, doc) -> str:
        """
        Extract text from document safely.
        """

        if hasattr(doc, "page_content"):
            return doc.page_content

        if hasattr(doc, "content"):
            return doc.content

        if hasattr(doc, "text"):
            return doc.text
        
        if isinstance(doc, dict):
            if "text" in doc:
                return doc["text"]

            if "content" in doc:
                return doc["content"]

            if "page_content" in doc:
                return doc["page_content"]

        logger.error("Invalid document schema: missing text field")
        logger.error(
            "Doc type: %s | Doc repr: %s",
            type(doc),
            str(doc)[:200],
        )
        raise ValueError(
            "Document has no valid text field (expected page_content/content/text)"
        )