import asyncio
import uuid

from app.graph.agent_graph_v2 import build_agent_graph_v2
from app.agents.state import AgentState
from scripts.test_rag_pipeline import build_rag_service


async def run_agent():

    # Initialize services
    rag_service = build_rag_service()

    # Build graph
    graph = build_agent_graph_v2(rag_service)

    # Create initial state
    state = AgentState(
        session_id=str(uuid.uuid4()),
        query="What is Apple's latest stock price?" #What does Apple's 10K say about revenue growth?"
    )

    # Run graph
    result = await graph.ainvoke(state)

    print("\n===== FINAL ANSWER =====\n")
    print(result["answer"])


if __name__ == "__main__":
    asyncio.run(run_agent())