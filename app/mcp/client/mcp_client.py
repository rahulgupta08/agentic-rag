from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters



class MCPClient:

    def __init__(self):

        self.server_params = StdioServerParameters(
            command="python",
            args=["-m", "app.mcp.server.mcp_server"]
        )

        self._session = None
        self._transport_cm = None

    async def connect(self):

        if self._session:
            return

        # open stdio transport
        self._transport_cm = stdio_client(self.server_params)
        read_stream, write_stream = await self._transport_cm.__aenter__()

        # open session
        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()

        await self._session.initialize()

    async def call_tool(self, tool_name, tool_input):

        await self.connect()

        result = await self._session.call_tool(tool_name, tool_input)

        return result.content

    async def list_tools(self):

        await self.connect()

        result = await self._session.list_tools()

        return result.tools
    
    async def close(self):

        if self._session:
            await self._session.__aexit__(None, None, None)
            self._session = None

        if self._transport_cm:
            await self._transport_cm.__aexit__(None, None, None)
            self._transport_cm = None