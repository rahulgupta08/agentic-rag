class AdvancedRAG:

    def __init__(self, retriever, threshold_filter=None, reranker=None):
        self.retriever = retriever
        self.threshold_filter = threshold_filter
        self.reranker = reranker

    def retrieve(self, query, top_k=5):

        docs = self.retriever.retrieve(query, top_k)

        if self.threshold_filter:
            docs = self.threshold_filter.filter(docs)

        if self.reranker:
            docs = self.reranker.rerank(query, docs)

        return docs