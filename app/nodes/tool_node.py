from app.agents.state import AgentState
from app.utils.debug import log_node_start, log_node_end



async def tool_node(state: AgentState):

    # Np tools yet
    log_node_start("tool", state)

    # Just pass state through

    if not state.tool_name:
        return state

    # Placeholder for tool execution
    tool_name = state.tool_name

    
    state.tool_output = f"Tool {tool_name} executed (placeholder)"

    log_node_end("tool", state)

    return state