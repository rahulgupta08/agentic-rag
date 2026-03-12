from app.agents.state import AgentState
from app.tools import tool_registry


def route_after_router(state: AgentState, tool_registry):

    plan = state.plan
    step = state.current_step

    print("\n🔀 Router Decision")
    print("Current Step:", step)
    print("Plan:", plan)

    if step >= len(plan):
        return "validator"

    action = plan[step]["tool"]

    next_node = tool_registry.get_node(action)

    print("Action:", action)
    print("Next Node:", next_node)
    print()

    if not next_node:
        return "generator"

    return next_node