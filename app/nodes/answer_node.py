from app.agents.state import AgentState
from app.services.rag_service import RAGService
from app.utils.debug import log_node_start, log_node_end



def create_answer_node(rag_service):

    async def answer_node(state: AgentState):

        log_node_start("answer", state)

        query = state.query
        context = state.context

        answer = await rag_service.generate_from_context(query, context)

        state.answer = answer

        log_node_end("answer", state)

        return state

    return answer_node