from accelerate import state

from app.agents.state import AgentState
from app.utils.debug import log_node_start, log_node_end


def create_executor_node(web_search_tool):

    async def executor_node(state: AgentState):

        log_node_start("executor", state)

        step = state.plan[state.current_step]

        query = step.get("input", {}).get("query", state.query)

        results = await web_search_tool.search(query)

        state.web_results = results

        state.current_step += 1

        log_node_end("executor", state)

        return state

    return executor_node