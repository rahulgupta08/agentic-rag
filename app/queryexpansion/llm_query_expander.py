class LLMQueryExpander:

    def __init__(self, llm):
        self.llm = llm

    def expand(self, query: str) -> str:

        prompt = f"""
                    You are a search query optimization assistant.

                    Rewrite the following user query to improve semantic retrieval.
                    Expand it with relevant synonyms and related financial/legal terminology.
                    Keep it concise and retrieval-focused.

                    Original Query:
                    {query}

                    Expanded Query:
                    """

        response = self.llm.invoke(prompt)

        expanded_query = response.content.strip()

        return expanded_query