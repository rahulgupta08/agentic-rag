from app.agents.state import AgentState
from app.utils.debug import log_node_start, log_node_end



async def validator_node(state: AgentState):
    log_node_start("validator", state)

    docs = state.retrieved_docs or []
    context = state.context or ""

    # Case 1 — no documents retrieved
    if not docs or len(docs) == 0:
        state.validation_score = 0.0
        state.needs_retry = False
        log_node_end("validator", state)
        return state

    # Case 2 — context empty
    if not context or len(context.strip()) == 0:
        state.validation_score = 0.0
        state.needs_retry = False
        log_node_end("validator", state)
        return state

    # Otherwise retrieval looks valid
    state.validation_score = 1.0
    state.needs_retry = False

    log_node_end("validator", state)

    return state