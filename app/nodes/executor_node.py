from app.agents.state import AgentState
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
        documents = normalize_tool_output(tool, result)

        metrics.log_retrieval(documents)


        new_results = state.tool_results + [{
            "tool": tool,
            "input": tool_input,
            "output": result,
            "documents": documents
        }]

        new_plan = state.plan[1:]

        return {
            "tool_results": new_results,
            "plan": new_plan
        }

    return executor_node

def normalize_tool_output(tool_name, raw_output):

    documents = []

    # Handle vector search
    if tool_name == "vector_search":

        for item in raw_output:

            text = getattr(item, "text", None)

            if text is None:
                text = str(item)

            documents.append({
                "text": text,
                "source": "vector_db",
                "tool":tool_name
            })

    # Handle web search
    elif tool_name == "web_search":

        for item in raw_output:

            # If MCP wrapped the result
            if hasattr(item, "text"):
                text = item.text

                documents.append({
                    "text": text,
                    "source": "web",
                    "tool":tool_name
                })

            # If it's a dict (fallback)
            elif isinstance(item, dict):

                title = item.get("title", "")
                snippet = item.get("snippet", "")
                url = item.get("url", "")

                text = f"{title} - {snippet}"

                documents.append({
                    "text": text,
                    "source": url,
                    "tool":tool_name
                })

            else:

                documents.append({
                    "text": str(item),
                    "source": "web",
                    "tool":tool_name
                })

    return documents    