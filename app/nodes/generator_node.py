from app.agents.state import AgentState
from app.core.observability import metrics


def create_generator_node(rag_service):

    async def generator_node(state: AgentState):

        query = state.query

        tool_results =  state.tool_results or []

        # Build context from retrieved docs
        context_parts = []

        for tool_result in tool_results:
            tool_name = tool_result.get("tool")
            output = tool_result.get("output", [])

            # Handle vector search results
            if tool_name == "vector_search":

                for doc in output:
                    if hasattr(doc, "page_content"):
                        context_parts.append(doc.page_content)
                    else:
                        context_parts.append(str(doc))

            # Handle web search results
            elif tool_name == "web_search":

                for result in output:
                    if isinstance(result, dict):
                        snippet = result.get("snippet", "")
                        context_parts.append(snippet)
                    else:
                        context_parts.append(str(result))

        context = "\n\n".join(context_parts)

        metrics.log_context_size(len(context_parts))

        # Call LLM
        answer = await rag_service.generate_from_context(
            query=query,
            context=context
            #history=conversation_history
        )

        

        return {
            "answer": answer.content,
            "plan": state.plan[1:]
        }

    return generator_node