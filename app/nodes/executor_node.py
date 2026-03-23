from app.agents.agent_state import AgentState
from app.mcp.client.mcp_singleton import mcp_client
from app.core.observability import metrics
import logging
logger = logging.getLogger(__name__)



def normalize(tool, result):

    if not result or result.get("status") != "success":
        return []

    data = result.get("results")

    #  search tool
    if tool == "search":
        return [
            {
                "text": item.get("content"),
                "source": item.get("source")
            }
            for item in data
        ]

    #  extract / summarize already structured
    if isinstance(data, list):
        return [
            {
                "text": item.get("content"),
                "source": item.get("source")
            }
            for item in data
        ]

    #  aggregate case
    if isinstance(data, dict) and "items" in data:
        return [
            {
                "text": item.get("content"),
                "source": item.get("source")
            }
            for item in data["items"]
        ]

    return []


def create_executor_node(rag_service):

    async def executor_node(state: AgentState):

        
        if not state.plan:
            return {}

        step = state.plan[0]

        tool = step["tool"]
        tool_input = step.get("input", {})

        metrics.log_tool(tool)

        # INTERNAL: vector search
        if tool == "vector_search":

            query = tool_input.get("query")
            logger.info(f"Vector Search query : {query}")

            documents = await rag_service.retriever.retrieve(query)

            result = []

            for doc in documents:
                text = doc.page_content if hasattr(doc, "page_content") else str(doc)

                result.append({
                    "text": text,
                    "source": "vector_db"
                })
            documents = result

        #  EXTERNAL: MCP tools
        else:
            logger.info(f"Tools just before calling MCP {tool}")
            if tool in {"extract", "summarize"}:

                previous_docs = []

                for r in state.tool_results:
                    previous_docs.extend(r.get("documents", []))

                tool_input = {
                    "data": previous_docs
            }
            result = await mcp_client.call_tool(tool, tool_input)
            # Normalize tool output
            documents = normalize(tool, result)

        
        

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
