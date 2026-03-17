from app.agents.agent_state import AgentState
from app.tools.tool_registry import tool_registry
from app.core.observability import metrics


def route_after_router(state: AgentState, tool_registry):

    if not state.plan:
        return "generator"

    
    step = state.plan[0]
    tool  = step["tool"]

    decision = tool_registry.get_node(tool)

    metrics.log_router_decision(decision)

    return decision