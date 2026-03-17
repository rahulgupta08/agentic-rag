from app.retrieval.base_retriever import BaseRetriever

class DenseRetriever(BaseRetriever):

    def __init__(self, vector_store, embedder, reranker=None):
        self.vector_store = vector_store
        self.embedder = embedder

    async def retrieve(self, query: str, top_k: int = 5):

        # Step 1 — Embed
        query_vector = await self.embedder.embed_query(query)

        # Step 2 — Retrieve broader set (recall stage)
        recall_k = max(top_k * 4, 20)  # configurable later

        documents =  await self.vector_store.dense_search(query_vector, recall_k)

        return documents