from app.agents.state import AgentState
from app.tools import tool_registry


def route_after_router(state: AgentState, tool_registry):

    if not state.plan:
        return "generator"

    print("\n🔀 Router Decision")
    print("Current Plan:", state.plan)


    step = state.plan[0]
    tool  = step["tool"]

    next_node = tool_registry.get_node(tool)

    print("Next Node:", next_node)

    return next_node