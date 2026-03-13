import sys, os
sys.path.append(os.getcwd())
from mcp.server.fastmcp import FastMCP
from app.mcp.server.tools.vector_search_tool import register_vector_search_tool
from app.mcp.server.tools.web_search_tool import register_web_search_tool


# Create MCP server
mcp = FastMCP("agent-tools")

async def show_tools():
    tools = await mcp.list_tools()


# Register tools
register_vector_search_tool(mcp)
register_web_search_tool(mcp)


if __name__ == "__main__":
    mcp.run()