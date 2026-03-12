from app.agents.state import AgentState
from app.utils.debug import log_node_start, log_node_end


def create_executor_node():

    async def executor_node(state: AgentState):

        log_node_start("executor", state)

        # Placeholder until web search tool is implemented
        # In future this will call MCP tools or web_search

        state.web_results = []

        # Advance execution step
        state.current_step += 1

        log_node_end("executor", state)

        return state

    return executor_node