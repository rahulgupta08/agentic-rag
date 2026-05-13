from app.bootstrap import bootstrap
bootstrap()
import logging
import asyncio
logger = logging.getLogger(__name__)
from app.agents.agent_state import AgentState
from app.core.observability import metrics
from app.contracts import RAGServiceProtocol

def create_generator_node(rag_service: RAGServiceProtocol):
    """
    Create a generator node that uses protocol interfaces instead of direct RAGService dependency.
    
    Args:
        rag_service: Protocol for RAG service operations
    """

    async def generator_node(state: AgentState):

        query = state.query
        tool_results =  state.tool_results or []

        # Build context from normalized documents
        context_parts = []

        for tool_result in tool_results:
           
            documents = tool_result.get("documents", [])

            for doc in documents:

                 # Case 1: normalized dict
                if isinstance(doc, dict):
                    text = doc.get("text")

                # Case 2: MCP TextContent object
                elif hasattr(doc, "text"):
                    text = doc.text

                else:
                    text = str(doc)

                if text:
                    context_parts.append(text)


        context = "\n\n".join(context_parts)
        metrics.log_context_size(len(context_parts))

        # Call LLM with timeout and error handling
        try:
            answer = await asyncio.wait_for(
                rag_service.generate_from_context(
                    query=query,
                    context=context
                ),
                timeout=60.0  # 60 second timeout for generation
            )
            
            logger.info(f"Generated Answer: {answer}")
            
            return {
                "answer": answer.content,
                "plan": state.plan[1:]
            }
            
        except asyncio.TimeoutError:
            logger.error("Generation timeout")
            return {
                "answer": "I apologize, but I'm taking too long to generate a response. Please try again.",
                "plan": state.plan[1:]
            }
            
        except Exception as e:
            logger.exception(f"Generation failed: {e}")
            return {
                "answer": "I encountered an error while generating the response. Please try again.",
                "plan": state.plan[1:]
            }

    return generator_node