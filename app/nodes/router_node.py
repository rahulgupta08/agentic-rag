from app.agents.state import AgentState
from app.utils.debug import log_node_start, log_node_end


def create_router_node(tool_registry):

    def router_node(state: AgentState):

        log_node_start("router", state)

        if not state.plan:
            decision = "validator"

        else:
            step = state.plan[0]
            tool = step.get("tool")

            decision = tool_registry.get_node(tool)

        print("\n🔀 Router Decision")
        print("Current Plan:", state.plan)
        print("Next Node:", decision)

        log_node_end("router", state)

        return {"next": decision}

    return router_node