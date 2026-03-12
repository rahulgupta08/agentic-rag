from accelerate import state

from app.agents.state import AgentState


from app.agents.state import AgentState
from app.utils.debug import log_node_start, log_node_end

def create_retriever_node(rag_service):

    async def retriever_node(state: AgentState):

        log_node_start("retriever", state)

        step = state.plan[state.current_step]

        query = step.get("input", {}).get("query", state.query)

        documents = await rag_service.retriever.retrieve(query)

        # Store retrieved docs
        state.retrieved_docs = documents

        # Build context string for generator
        context_parts = []

        for doc in documents:
            if hasattr(doc, "page_content"):
                context_parts.append(doc.page_content)
            else:
                context_parts.append(str(doc))

        state.context = "\n\n".join(context_parts)

        # IMPORTANT: increment execution step
        state.current_step += 1

        log_node_end("retriever", state)

        return state

    return retriever_node