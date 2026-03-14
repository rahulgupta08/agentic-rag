from app.agents.state import AgentState
from app.core.observability import metrics
from app.logging.logger import logger



def create_generator_node(rag_service):

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

        # Call LLM
        answer = await rag_service.generate_from_context(
            query=query,
            context=context
            #history=conversation_history
        )

        logger.info(f"Generated Answer  {answer}")

        return {
            "answer": answer.content,
            "plan": state.plan[1:]
        }

    return generator_node