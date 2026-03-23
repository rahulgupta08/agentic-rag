import httpx


class MCPClient:

    def __init__(self):
        self.base_url = "http://localhost:8000"  # 🔥 change if needed

        self.tool_endpoint_map = {
            "search": "/tools/search",
            "extract": "/tools/extract",
            "summarize": "/tools/summarize",
            "aggregate": "/tools/aggregate",
        }

        self.client = httpx.AsyncClient(timeout=10.0)

    async def call_tool(self, tool_name, tool_input):

        endpoint = self.tool_endpoint_map.get(tool_name)

        if not endpoint:
            raise ValueError(f"Unknown tool: {tool_name}")

        url = f"{self.base_url}{endpoint}"

        response = await self.client.post(url, json=tool_input)

        response.raise_for_status()

        data = response.json()

        return data

    async def list_tools(self):

        url = f"{self.base_url}/tools"

        response = await self.client.get(url)
        response.raise_for_status()

        data = response.json()

        return data["tools"]

    async def close(self):
        await self.client.aclose()