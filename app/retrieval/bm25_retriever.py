from rank_bm25 import BM25Okapi
from app.retrieval.base_retriever import BaseRetriever


class BM25Retriever(BaseRetriever):

    def __init__(self, documents: list[str]):
        self.documents = documents
        self.tokenized = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized)

    async def retrieve(self, query: str, top_k: int = 5):

        scores = self.bm25.get_scores(query.split())

        ranked = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {
                "text": doc,
                "score": float(score),  # unified key
                "metadata": {}
            }
            for doc, score in ranked[:top_k]
        ]