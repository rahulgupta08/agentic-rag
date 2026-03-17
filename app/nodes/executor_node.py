from app.agents.agent_state import AgentState
from app.mcp.client.mcp_singleton import mcp_client
from app.core.observability import metrics



def create_executor_node():

    async def executor_node(state: AgentState):

        
        if not state.plan:
            return {}

        step = state.plan[0]

        tool = step["tool"]
        tool_input = step.get("input", {})

        metrics.log_tool(tool)

        result = await mcp_client.call_tool(tool, tool_input)

        # Normalize tool output
        documents = result

        metrics.log_retrieval(documents)


        new_results = state.tool_results + [{
            "tool": tool,
            "input": tool_input,
            "documents": documents
        }]

        new_plan = state.plan[1:]

        return {
            "tool_results": new_results,
            "plan": new_plan
        }

    return executor_node
