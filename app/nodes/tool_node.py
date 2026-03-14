from app.agents.state import AgentState




async def tool_node(state: AgentState):


    # Just pass state through

    if not state.tool_name:
        return state

    # Placeholder for tool execution
    tool_name = state.tool_name

    
    state.tool_output = f"Tool {tool_name} executed (placeholder)"

    return state