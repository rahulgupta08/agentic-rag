from app.agents.state import AgentState
from app.utils.debug import log_node_start, log_node_end


def planner_node(state: AgentState):

    log_node_start("planner", state)

    plan = "retrieve"

    if state.metadata is None:
        state.metadata = {}

    state.metadata["plan"] = plan

    log_node_start("planner", state)

    return state