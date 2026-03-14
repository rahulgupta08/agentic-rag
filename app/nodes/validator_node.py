from app.agents.state import AgentState
from app.core.observability import metrics



async def validator_node(state: AgentState):

    vector_docs = []

    # Extract vector_search results
    for result in state.tool_results:
        if result["tool"] == "vector_search":
            vector_docs = result.get("output", [])

    # Case 1 — no documents retrieved
    if not vector_docs or len(vector_docs) == 0:

        metrics.log_validation(0.0)
        

        retry_count = state.retry_count + 1

        metrics.log_retry(retry_count)
        
        if retry_count > state.max_retries:

            return {
                "validation_score": 0.0,
                "needs_retry": False
            }

        
        
        return {
            "validation_score": 0.0,
            "needs_retry": True,
            "retry_count" : retry_count
        }

    metrics.log_validation(1.0)
    return {
        "validation_score": 1.0,
        "needs_retry": False
    }