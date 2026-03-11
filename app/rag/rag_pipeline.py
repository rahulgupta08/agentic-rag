import asyncio
import logging
from typing import Dict


class RAGPipeline:
    """
    Async wrapper around RAGService.

    This allows the existing synchronous RAGService to run
    inside an async environment (required later for LangGraph).
    """

    def __init__(self, rag_service):
        self.rag_service = rag_service


    async def run(self, query: str, top_k: int = 5) -> Dict:
        """
        Run the RAG pipeline asynchronously.

        Steps:
        1. Validate query
        2. Retrieve documents
        3. Build prompt
        4. Generate answer
        5. Return structured response
        """

        print("Running RAG pipeline for query: ", query)

        # Run the synchronous RAGService.generate() inside a thread
        result = await asyncio.to_thread(
            self.rag_service.generate,
            query,
            top_k
        )

        print("RAG pipeline completed")

        return result