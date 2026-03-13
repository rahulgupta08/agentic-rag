from app.agents.state import AgentState
from app.utils.debug import log_node_start, log_node_end


def router_node(state: AgentState):

    log_node_start("router", state)

    if not state.plan:
        decision = "validator"
    else:
        step = state.plan[0]
        tool = step.get("tool")

        if tool == "generate":
            decision = "generator"
        else:
            decision = "executor"

    print("\n🔀 Router Decision")
    print("Current Plan:", state.plan)
    print("Next Node:", decision)

    log_node_end("router", state)

    return {"next": decision}