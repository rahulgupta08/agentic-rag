from app.bootstrap import bootstrap
bootstrap()
import logging
logger = logging.getLogger(__name__)
import asyncio
from typing import Dict


class RAGPipeline:
    """
    Pipeline for RAG operations.
    Uses synchronous operations for better performance in RAG mode.
    """

    def __init__(self, rag_service):
        self.rag_service = rag_service

    def run(self, query: str, top_k: int = 5) -> Dict:
        """
        Run the RAG pipeline synchronously.

        Steps:
        1. Validate query
        2. Retrieve documents
        3. Build prompt
        4. Generate answer
        5. Return structured response
        """

        logger.info(f"Running RAG pipeline for query: {query}")

        # Run the synchronous RAGService.generate()
        result = self.rag_service.generate(query, top_k)

        logger.info("RAG pipeline completed")

        return result

    async def run_agent(self, query: str, top_k: int = 5) -> Dict:
        """
        Run the pipeline asynchronously for agent mode.

        Steps:
        1. Validate query
        2. Retrieve documents
        3. Build prompt
        4. Generate answer
        5. Return structured response
        """

        logger.info(f"Running agent RAG pipeline for query: {query}")

        # Run the async RAGService.agenerate()
        result = await self.rag_service.agenerate(query, top_k)

        logger.info("Agent RAG pipeline completed")

        return result