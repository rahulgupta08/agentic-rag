from app.agents.state import AgentState
from app.utils.debug import log_node_start, log_node_end


def create_query_classifier_node(llm):

    async def query_classifier_node(state: AgentState):

        log_node_start("query_classifier", state)

        query = state.query

        prompt = f"""
                You are an AI system that decides whether a query should use internal documents or internet search.

                The internal knowledge base contains:
                - Apple 10K reports
                - company financial filings
                - SEC filings
                - corporate reports

                Classification rules:

                rag_query:
                Use if the question refers to company filings, financial reports, or internal documents.

                web_query:
                Use if the question requires:
                - current stock prices
                - latest news
                - current events
                - real-time information

                Return ONLY:
                rag_query
                or
                web_query

                Query:
                {query}
                """

        response = await llm.ainvoke(prompt)

        label = response.content.strip().lower()

        if "web_query" in label:
            state.query_type = "web_query"
        else:
            state.query_type = "rag_query"

        log_node_end("query_classifier", state)

        return state

    return query_classifier_node