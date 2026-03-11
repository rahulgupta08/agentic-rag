from app.retrieval.base_retriever import BaseRetriever

class DenseRetriever(BaseRetriever):

    def __init__(self, vector_store, embedder, reranker=None):
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int = 5):

        # Step 1 — Embed
        query_vector = self.embedder.embed_query(query)

        # Step 2 — Retrieve broader set (recall stage)
        documents =  self.vector_store.dense_search(query_vector, top_k)

        if self.reranker:
             documents = self.reranker.rerank(
                 query,
                 documents,
                 top_k= top_k

             )
        else:
            documents = documents[:top_k]

        

        return documents