from app.agents.state import AgentState
from app.utils.debug import log_node_start, log_node_end


def create_generator_node(rag_service):

    async def generator_node(state: AgentState):

        log_node_start("generator", state)

        query = state.query

        retrieved_docs = state.retrieved_docs or []
        web_results = state.web_results or []
        conversation_history = state.conversation_history or []

        # Build context from retrieved docs
        context_parts = []

        for doc in retrieved_docs:
            if hasattr(doc, "page_content"):
                context_parts.append(doc.page_content)
            else:
                context_parts.append(str(doc))

        # Add web results if available
        for result in web_results:
            if isinstance(result, dict):
                snippet = result.get("snippet", "")
                context_parts.append(snippet)
            else:
                context_parts.append(str(result))

        context = "\n\n".join(context_parts)

        # Call LLM
        answer = await rag_service.generate_from_context(
            query=query,
            context=context
            #history=conversation_history
        )

        state.answer = answer.content

        # Increment execution step
        state.current_step += 1

        log_node_end("generator", state)

        return state

    return generator_node