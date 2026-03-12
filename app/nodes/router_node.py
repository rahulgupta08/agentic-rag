from app.agents.state import AgentState
from app.utils.debug import log_node_start, log_node_end


MAX_STEPS = 10


def router_node(state: AgentState):

    log_node_start("router", state)

    log_node_end("rounter", state)
    return state