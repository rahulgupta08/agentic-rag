from app.agents.state import AgentState
from app.utils.debug import log_node_start, log_node_end
from app.mcp.client.mcp_singleton import mcp_client


def create_executor_node():

    async def executor_node(state: AgentState):

        log_node_start("executor", state)

        if not state.plan:
            log_node_end("executor", state)
            return {}

        step = state.plan[0]

        tool = step["tool"]
        tool_input = step.get("input", {})

        result = await mcp_client.call_tool(tool, tool_input)

        new_results = state.tool_results + [{
            "tool": tool,
            "input": tool_input,
            "output": result
        }]

        new_plan = state.plan[1:]

        log_node_end("executor", state)

        return {
            "tool_results": new_results,
            "plan": new_plan
        }

    return executor_node