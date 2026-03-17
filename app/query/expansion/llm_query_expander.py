from app.query.base_query_transformer import BaseQueryTransformer


class LLMQueryExpander(BaseQueryTransformer):

    def __init__(self, llm, num_queries=3):
        self.llm = llm
        self.num_queries = num_queries

    async def transform(self, query: str):

        prompt = f"""
                Generate {self.num_queries} different search queries.

                Query: {query}
                Queries:
                """

        response = await self.llm.ainvoke(prompt)

        queries = [
            q.strip("- ").strip()
            for q in response.content.split("\n")
            if q.strip()
        ]

        return queries[:self.num_queries]