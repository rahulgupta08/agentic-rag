from langgraph.graph import StateGraph, END

from app.agents.state import AgentState

from app.core.memory_reader import memory_reader_node
from app.core.memory_writer import memory_writer_node

from app.nodes.query_classifier_node import query_classifier_node
from app.nodes.planner_node import planner_node
from app.nodes.router_node import router_node
from app.nodes.retriever_node import create_retriever_node
from app.nodes.executor_node import create_executor_node
from app.nodes.generator_node import create_generator_node

from app.nodes.validator_node import validator_node
from app.graph.router_logic import route_after_router


def build_agent_graph_v2(rag_service):

    graph = StateGraph(AgentState)

    # Inject dependencies
    retriever_node = create_retriever_node(rag_service)
    generator_node = create_generator_node(rag_service)
    executor_node = create_executor_node()
    

    # Register nodes
    graph.add_node("memory_reader", memory_reader_node)
    graph.add_node("query_classifier", query_classifier_node)
    graph.add_node("planner", planner_node)
    graph.add_node("router", router_node)

    graph.add_node("retriever", retriever_node)
    graph.add_node("executor", executor_node)
    graph.add_node("generator", generator_node)

    graph.add_node("validator", validator_node)
    graph.add_node("memory_writer", memory_writer_node)

    # Entry point
    graph.set_entry_point("memory_reader")

    # Linear flow
    graph.add_edge("memory_reader", "query_classifier")
    graph.add_edge("query_classifier", "planner")
    graph.add_edge("planner", "router")

    # Router dynamic execution
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retriever": "retriever",
            "executor": "executor",
            "generator": "generator",
            "validator": "validator",
        },
    )

    # Loop back to router
    graph.add_edge("retriever", "router")
    graph.add_edge("executor", "router")

    # Final generation
    graph.add_edge("generator", "validator")

    # Memory persistence
    graph.add_edge("validator", "memory_writer")

    # End
    graph.add_edge("memory_writer", END)

    return graph.compile()