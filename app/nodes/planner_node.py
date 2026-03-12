from app.agents.state import AgentState
from app.utils.debug import log_node_start, log_node_end


def planner_node(state: AgentState):

    log_node_start("planner", state)

    query_type = state.query_type or "rag_query"

    # Determine execution plan
    if query_type == "rag_query":
        plan = ["vector_search", "generate"]

    elif query_type == "web_query":
        plan = ["web_search", "generate"]

    else:
        # fallback
        plan = ["generate"]

    log_node_end("planner", plan)

    return {
        "plan": plan,
        "current_step": 0
    }

    