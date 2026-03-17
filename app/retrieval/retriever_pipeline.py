from app.retrieval.base_retriever import BaseRetriever


class RetrieverPipeline(BaseRetriever):

    def __init__(
        self,
        base_retriever,
        reranker=None,
        query_transformer=None,
        logger=None
    ):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.query_transformer = query_transformer
        self.logger = logger

    async def retrieve(self, query: str, top_k: int = 5):

        original_query = query

        # Step 1 — Query transform
        if self.query_transformer:
            query = await self.query_transformer.transform(query)

        # Step 2 — Recall
        documents = await self.base_retriever.retrieve(query, top_k)

        # Step 3 — Rerank
        if self.reranker:
            documents = await self.reranker.rerank(
                query,
                documents,
                top_k=top_k
            )
        else:
            documents = documents[:top_k]

        # Step 4 — Logging
        if self.logger:
            for i, doc in enumerate(documents):
                score = doc.metadata.get("reranker_score", None)
                await self.logger.log(f"[RERANKED {i}] score={score} | {doc.page_content[:80]}")

        return documents