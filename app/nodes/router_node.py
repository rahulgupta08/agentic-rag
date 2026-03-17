from app.agents.agent_state import AgentState
from app.core.observability import metrics


def create_router_node(tool_registry):

    def router_node(state: AgentState):

        if not state.plan:
            decision = "validator"

        else:
            step = state.plan[0]
            tool = step.get("tool")

            decision = tool_registry.get_node(tool)
        
        metrics.log_router_decision(decision)

        return {"next": decision}

    return router_node