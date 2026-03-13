from app.agents.state import AgentState
from app.utils.debug import log_node_start, log_node_end


async def validator_node(state: AgentState):

    log_node_start("validator", state)

    vector_docs = []

    # Extract vector_search results
    for result in state.tool_results:
        if result["tool"] == "vector_search":
            vector_docs = result.get("output", [])

    # Case 1 — no documents retrieved
    if not vector_docs or len(vector_docs) == 0:

        retry_count = state.retry_count + 1
        
        if retry_count > state.max_retries:
            log_node_end("validator", state)

            return {
                "validation_score": 0.0,
                "needs_retry": False
            }

        log_node_end("validator", state)

        return {
            "validation_score": 0.0,
            "needs_retry": True,
            "retry_count" : retry_count
        }

    # Case 2 — retrieval looks valid
    log_node_end("validator", state)

    return {
        "validation_score": 1.0,
        "needs_retry": False
    }