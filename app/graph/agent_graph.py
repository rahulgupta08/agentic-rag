from langgraph.graph import StateGraph, END

from app.agents import agent_state
from app.agents.agent_state import AgentState

from app.core.memory_reader import memory_reader_node
from app.core.memory_writer import memory_writer_node
from app.nodes.planner_node import create_planner_node
from app.nodes.router_node import create_router_node
from app.nodes.executor_node import create_executor_node
from app.nodes.generator_node import create_generator_node


from app.nodes.validator_node import create_validator_node
from app.tools.tool_registry import ToolRegistry
from app.mcp.client.mcp_singleton import mcp_client


async def build_agent_graph(rag_service):

    tool_registry = ToolRegistry()

    tools = await mcp_client.list_tools()

    for tool in tools:
            tool_registry.register(
                tool["name"],
                tool["description"],
                "executor",
                capabilities=["retrieval"]
            )

    tool_registry.register(
        "vector_search",
        "Search internal knowledge base",
        "executor",
        capabilities=["retrieval"]
    )


    tool_registry.register(
        "generate",
        "Generate the final answer using available context",
        "generator",
        capabilities=["generation"]

    )
    

    graph = StateGraph(AgentState)
    llm = rag_service.llm
    # Inject dependencies
    generator_node = create_generator_node(rag_service)

    executor_node = create_executor_node(rag_service)
    planner_node = create_planner_node(llm, tool_registry)
    router_node = create_router_node(tool_registry)
    validator_node = create_validator_node(tool_registry)
    

    # Register nodes
    graph.add_node("memory_reader", memory_reader_node)
    graph.add_node("planner", planner_node)
    graph.add_node("router", router_node)

    
    graph.add_node("executor", executor_node)
    graph.add_node("generator", generator_node)

    graph.add_node("validator", validator_node)
    graph.add_node("memory_writer", memory_writer_node)


    # Entry point
    graph.set_entry_point("memory_reader")


    # Linear flow
    graph.add_edge("memory_reader", "planner")
    graph.add_edge("planner", "router")


    graph.add_conditional_edges(
        "router",
        lambda state: state.next,
        {
            "executor": "executor",
            "generator": "generator",
            "validator": "validator",
        },
    )

    # Loop back to router
    graph.add_edge("executor", "router")

    # Final generation
    graph.add_edge("generator", "validator")

    # Memory persistence
    graph.add_conditional_edges(
        "validator",
        lambda state: "planner" if state.needs_retry else "memory_writer",
        {
            "planner": "planner",
            "memory_writer": "memory_writer",
        },
    )

    # End
    graph.add_edge("memory_writer", END)

    return graph.compile()