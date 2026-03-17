from app.retrieval.base_retriever import BaseRetriever

class DenseRetriever(BaseRetriever):

    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    async def retrieve(self, query: str, top_k: int = 5):

        query_vector = await self.embedder.embed_query(query)

        recall_k = max(top_k * 4, 20)

        docs = await self.vector_store.dense_search(query_vector, recall_k)

        results = []
        for doc in docs:

            if isinstance(doc, dict):
                text = doc.get("text", "")
                score = doc.get("score", 0)
                metadata = doc.get("metadata", {})
            else:
                text = doc.page_content
                score = getattr(doc, "score", 0)
                metadata = getattr(doc, "metadata", {})

            results.append({
                "text": text,
                "score": float(score),
                "metadata": metadata
            })

        return results