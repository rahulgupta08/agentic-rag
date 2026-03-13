from ddgs import DDGS


def register_web_search_tool(mcp):

    @mcp.tool()
    async def web_search(query: str):

        results = []

        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append({
                    "title": r.get("title"),
                    "snippet": r.get("body"),
                    "url": r.get("href")
                })

        return results