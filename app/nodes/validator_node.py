from app.agents.state import AgentState
from app.utils.debug import log_node_start, log_node_end



async def validator_node(state: AgentState):
    log_node_start("validator", state)

    context = state.context
    docs = state.retrieved_docs

    # Case 1 — no documents retrieved
    if not docs or len(docs) == 0:
        state.is_valid = False
        state.retry_count += 1
        return state

    # Case 2 — context empty
    if not context or len(context.strip()) == 0:
        state.is_valid = False
        state.retry_count += 1
        return state

    # Otherwise retrieval is valid
    state.is_valid = True

    log_node_end("validator", state)

    return state