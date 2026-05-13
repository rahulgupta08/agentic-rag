from app.retrieval.base_retriever import BaseRetriever
import logging
logger = logging.getLogger(__name__)

class DenseRetriever(BaseRetriever):

    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5):
        """Synchronous retrieve for RAG mode"""
        # Use sync embed_query
        query_vector = self.embedder.embed_query(query)

        recall_k = max(top_k * 4, 20)

        # Use sync dense_search
        docs = self.vector_store.dense_search(query_vector, recall_k)
        return self._process_docs(docs)

    async def aretrieve(self, query: str, top_k: int = 5):
        """Asynchronous retrieve for Agent mode"""
        # Use async embed_query_async
        query_vector = await self.embedder.embed_query_async(query)

        recall_k = max(top_k * 4, 20)

        # Use async dense_search
        docs = await self.vector_store.dense_search_async(query_vector, recall_k)
        return self._process_docs(docs)

    def _process_docs(self, docs):
        """Helper method to process and format retrieved documents"""
        results = []
        for doc in docs:
            if isinstance(doc, dict):
                text = doc.get("text", "")
                id = doc.get("id", "")
                score = doc.get("score", 0)
                metadata = doc.get("metadata", {})
            else:
                text = doc.page_content
                score = getattr(doc, "score", 0)
                metadata = getattr(doc, "metadata", {})
                id = getattr(doc, "id", "")

            results.append({
                "text": text,
                "id": id,
                "score": float(score),
                "metadata": metadata
            })

        return results