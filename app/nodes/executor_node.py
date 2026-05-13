from app.agents.agent_state import AgentState
from app.core.observability import metrics
import logging
import asyncio
from app.core.exceptions import RetrievalError
from app.contracts import RetrieverProtocol, LLMProtocol
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


def create_executor_node(retriever: RetrieverProtocol, llm: LLMProtocol, mcp_client):
    """
    Create an executor node that uses protocol interfaces instead of direct RAGService dependency.
    
    Args:
        retriever: Protocol for document retrieval
        llm: Protocol for LLM operations
        mcp_client: MCP client for external tool calls
    """

    async def executor_node(state: AgentState):
        if not state.plan:
            return {}

        step = state.plan[0]
        tool = step["tool"]
        tool_input = step.get("input", {})

        metrics.log_tool(tool)

        documents = []
        error = None

        try:
            # INTERNAL: vector search
            if tool == "vector_search":
                query = tool_input.get("query")
                logger.info(f"Vector Search query: {query}")

                try:
                    raw_documents = await asyncio.wait_for(
                        retriever.retrieve(query),
                        timeout=10.0  # 10 second timeout
                    )

                    for doc in raw_documents:
                        text = doc.page_content if hasattr(doc, "page_content") else str(doc)
                        documents.append({
                            "text": text,
                            "source": "vector_db"
                        })

                except asyncio.TimeoutError:
                    logger.error(f"Vector search timeout for query: {query}")
                    error = "timeout"
                except Exception as e:
                    logger.error(f"Vector search failed: {e}")
                    error = str(e)
            
            # EXTERNAL: MCP tools
            else:
                logger.info(f"Calling MCP tool: {tool}")

                if tool in {"extract", "summarize"}:
                    previous_docs = []
                    for r in state.tool_results:
                        previous_docs.extend(r.get("documents", []))
                    tool_input = {"data": previous_docs}
                
                try:
                    result = await asyncio.wait_for(
                        mcp_client.call_tool(tool, tool_input),
                        timeout=30.0  # 30 second timeout for external calls
                    )
                    documents = normalize(tool, result)
                    
                except asyncio.TimeoutError:
                    logger.error(f"MCP tool timeout: {tool}")
                    error = "timeout"
                except ValueError as e:
                    logger.error(f"Unknown MCP tool: {tool}")
                    error = f"unknown_tool: {tool}"
                except Exception as e:
                    logger.error(f"MCP call failed for {tool}: {e}")
                    error = str(e)
        
        except Exception as e:
            logger.exception(f"Executor node failed: {e}")
            error = str(e)
        
        metrics.log_retrieval(documents)

        new_results = state.tool_results + [{
            "tool": tool,
            "input": tool_input,
            "documents": documents,
            "error": error
        }]

        new_plan = state.plan[1:]

        return {
            "tool_results": new_results,
            "plan": new_plan
        }

    return executor_node
