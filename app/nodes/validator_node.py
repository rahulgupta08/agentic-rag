from app.agents.state import AgentState
from app.core.observability import metrics



def create_validator_node(tool_registry):

    async def validator_node(state: AgentState):

        retrieved_docs = []

        for result in state.tool_results:

            tool_name = result["tool"]

            if tool_registry.has_capability(tool_name, "retrieval"):
                retrieved_docs.extend(result.get("documents", []))


        # Case 1 — no documents retrieved
        if not retrieved_docs:

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
                "retry_count": retry_count
            }

        metrics.log_validation(1.0)

        return {
            "validation_score": 1.0,
            "needs_retry": False
        }

    return validator_node