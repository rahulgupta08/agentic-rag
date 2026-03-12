import asyncio
from ddgs import DDGS


class WebSearchTool:

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    async def search(self, query: str):
        print(f"Performing ddgs search for query: {query}")
        def _search():
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=self.max_results):
                    results.append({
                        "title": r.get("title"),
                        "snippet": r.get("body"),
                        "url": r.get("href")
                    })
            return results

        return await asyncio.to_thread(_search)