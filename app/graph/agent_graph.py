from langgraph.graph import StateGraph, END

from app.agents.state import AgentState

from app.nodes.planner_node import planner_node
from app.nodes.retriever_node import create_retriever_node
from app.nodes.tool_node import tool_node
from app.nodes.validator_node import validator_node
from app.nodes.answer_node import create_answer_node
from dotenv import load_dotenv

load_dotenv()

def route_after_validation(state: AgentState):

    if state.is_valid:
        return "answer"

    if state.retry_count >= 2:
        return "answer"

    return "retriever"

def build_agent_graph(rag_service):

    graph = StateGraph(AgentState)

    # Inject dependency
    answer_node = create_answer_node(rag_service)
    retriever_node = create_retriever_node(rag_service)

    # Register nodes
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("tool", tool_node)
    graph.add_node("validator", validator_node)
    graph.add_node("answer", answer_node)

    # Entry point
    graph.set_entry_point("planner")

    # Standard edges
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "tool")
    graph.add_edge("tool", "validator")

    # Conditional routing
    graph.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "answer": "answer",
            "retriever": "retriever",
        }
    )

    # End
    graph.add_edge("answer", END)

    return graph.compile()