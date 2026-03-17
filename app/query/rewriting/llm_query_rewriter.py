from app.query.base_query_transformer import BaseQueryTransformer


class LLMQueryRewriter(BaseQueryTransformer):

    def __init__(self, llm):
        self.llm = llm

    async def transform(self, query: str):

        prompt = f"""
                Rewrite the query to improve retrieval quality.
                Keep intent unchanged.

                Query: {query}
                Rewritten:
                """

        response = await self.llm.ainvoke(prompt)
        rewritten = response.content.strip()

        return [rewritten]