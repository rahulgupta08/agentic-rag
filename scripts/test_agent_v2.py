from app.bootstrap import bootstrap
bootstrap()
import logging
logger = logging.getLogger(__name__)
import asyncio
import uuid

from app.graph.agent_graph_v2 import build_agent_graph_v2
from app.agents.state import AgentState
from scripts.test_rag_pipeline import build_rag_service
from app.mcp.client.mcp_singleton import mcp_client



async def run_agent():

    logger.info("Running Graph Agent")

    # Initialize services
    rag_service = build_rag_service()

    # Build graph
    graph = await build_agent_graph_v2(rag_service)

    # Create initial state
    state = AgentState(
        session_id=str(uuid.uuid4()),
        query= "what is the Apple's latest stock price?"
        #"What is Apple's latest stock price?" #What does Apple's 10K say about revenue growth?"
       # "What does Apple's 2024 10K say about revenue growth with its latest stock price?"
    )

    # Run graph
    result = await graph.ainvoke(state)

    logger.info("===== FINAL ANSWER =====")
    logger.info(result["answer"])
    # CLOSE VECTOR DB CLIENT
    if hasattr(rag_service, "vector_store"):
        client = getattr(rag_service.vector_store, "client", None)
        if client:
            client.close()

    await mcp_client.close()

if __name__ == "__main__":
    asyncio.run(run_agent())