from app.agents.agent_state import AgentState
from app.core.observability import metrics
from app.rerankers.llm_reranker import LLMReranker


def create_validator_node(tool_registry, llm=None):
    # Initialize reranker if LLM provided
    reranker = LLMReranker(llm) if llm else None
    
    async def validator_node(state: AgentState):
        retrieved_docs = []
        
        for result in state.tool_results:
            tool_name = result["tool"]
            if tool_registry.has_capability(tool_name, "retrieval"):
                retrieved_docs.extend(result.get("documents", []))
        
        # Case 1 — no documents retrieved
        if not retrieved_docs:
            metrics.log_validation(0.0)
            return _handle_retry(state)
        
        # Case 2 — documents exist, check relevance
        if reranker:
            try:
                # Score documents for relevance
                scored_docs = await reranker.rerank(
                    state.query,
                    retrieved_docs,
                    top_k=len(retrieved_docs)
                )
                
                # Calculate average relevance score
                avg_score = sum(
                    doc.get("reranker_score", 0) 
                    for doc in scored_docs
                ) / len(scored_docs)
                
                # Threshold for acceptable relevance
                if avg_score < 0.3:
                    print(f"Low relevance score: {avg_score:.2f}, triggering retry")
                    metrics.log_validation(avg_score)
                    return _handle_retry(state)
                
                metrics.log_validation(avg_score)
                return {
                    "validation_score": avg_score,
                    "needs_retry": False
                }
                
            except Exception as e:
                print(f"Validation scoring failed: {e}")
                # Fall back to simple existence check
        
        # Fallback: documents exist, assume valid
        metrics.log_validation(1.0)
        return {
            "validation_score": 1.0,
            "needs_retry": False
        }
    
    def _handle_retry(state):
        """Handle retry logic."""
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
    
    return validator_node