from app.retrieval.base_retriever import BaseRetriever

class HybridRetriever(BaseRetriever):

    def __init__(self, dense_retriever, bm25_retriever, fusion):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.fusion = fusion

    async def retrieve(self, query: str, top_k: int = 5):

        recall_k = max(top_k * 4, 20)

        dense_docs = await self.dense.retrieve(query, recall_k)
        bm25_docs = await self.bm25.retrieve(query, recall_k)

        return self.fusion.merge(dense_docs, bm25_docs, top_k)