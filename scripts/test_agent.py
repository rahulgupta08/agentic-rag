import asyncio

from app.graph.agent_graph import build_agent_graph
from app.agents.state import AgentState
from test_rag_pipeline import build_rag_service


async def run_agent():

    rag_service = build_rag_service()

    graph = build_agent_graph(rag_service)

    state = AgentState(
        query="What was Apple's revenue in 2024?"
    )

    result = await graph.ainvoke(state)

    print(result["answer"])


asyncio.run(run_agent())