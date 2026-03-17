from app.retrieval.base_retriever import BaseRetriever


class RetrieverPipeline(BaseRetriever):

    def __init__(
        self,
        base_retriever,
        reranker=None,
        query_transformer=None,
        post_processors=None,
        logger=None
    ):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.query_transformer = query_transformer
        self.post_processors = post_processors or []
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

        for processor in self.post_processors:
            docs = processor.process(docs)


        return documents