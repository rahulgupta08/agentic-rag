from app.retrieval.base_retriever import BaseRetriever

class HybridRetriever(BaseRetriever):

    def __init__(self, store, alpha=0.5):
        self.store = store
        self.alpha = alpha

    def retrieve(self, query: str, top_k: int = 5):
        return self.store.hybrid_search(query, top_k, self.alpha) 