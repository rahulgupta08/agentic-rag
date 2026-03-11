from app.nodes.planner_node import planner_node
from app.agents.state import AgentState
from app.nodes.tool_node import tool_node
from app.nodes.validator_node import validator_node
from app.nodes.answer_node import create_answer_node
from test_rag_pipeline import build_rag_service
import asyncio

async def test_answer_node():
    state = AgentState(
        query="what is Apple revenue 2024",
        context="Apple reported revenue of $383 billion in 2024."
    )

    rag_service = build_rag_service()

    answer_node =  create_answer_node(rag_service)
    result = await answer_node(state)


    print(result.answer)
    


asyncio.run(test_answer_node())

