from app.agents.state import AgentState


def query_classifier_node(state: AgentState):

    query = state.query.lower()

    # Simple keyword based classification

    web_keywords = [
        "current",
        "latest",
        "today",
        "news",
        "stock price",
        "recent",
        "now"
    ]

    # Detect if query requires web search
    if any(keyword in query for keyword in web_keywords):
        query_type = "web_query"

    # Default assumption: RAG query
    else:
        query_type = "rag_query"

    return {
        "query_type": query_type
    }