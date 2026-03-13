import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.graph.agent_graph_v2 import build_agent_graph_v2
from scripts.test_rag_pipeline import build_rag_service
import asyncio

async def main():

    rag_service = build_rag_service()
    graph = await build_agent_graph_v2(rag_service)
    png_bytes =  graph.get_graph().draw_mermaid_png()

    with open("data/graph_with_mcp.png", "wb") as f:
        f.write(png_bytes)


    print(graph.get_graph().draw_mermaid())


asyncio.run(main())
