import asyncio

class QueryEmbedder:

    def __init__(self, model):
        self.model = model

    async def embed_query(self, query: str):

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.model.encode,
            query
        )