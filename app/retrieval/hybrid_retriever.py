from app.retrieval.base_retriever import BaseRetriever

class HybridRetriever(BaseRetriever):

    def __init__(self, store, alpha=0.5):
        self.store = store
        self.alpha = alpha

    async def retrieve(self, query: str, top_k: int = 5):
        recall_k = max(top_k * 4, 20)

        return await self.store.hybrid_search(query, recall_k, self.alpha) 