
class RAGPromptBuilder:

    def __init__(
        self,
        enforce_citation: bool = True,
        bullet_points: bool = False,
        strict_mode: bool = True,
        internal_reasoning: bool = True
    ):
        self.enforce_citation = enforce_citation
        self.bullet_points = bullet_points
        self.strict_mode = strict_mode
        self.internal_reasoning = internal_reasoning


    def build_generation_prompt(
        self,
        context: str,
        question: str
    ) -> str:

        reasoning_instruction = ""
        citation_instruction = ""
        strict_instruction = ""

        if self.internal_reasoning:
            reasoning_instruction = (
                "First analyze the context carefully and reason step-by-step internally. "
                "Do not expose your reasoning.\n"
            )

        if self.enforce_citation:
            citation_instruction = (
                "Every factual statement MUST include citation in format: "
                "[Document ID: <id>]\n"
            )

        if self.strict_mode:
            strict_instruction = (
                "If the answer is not fully supported by the context, "
                "respond with: "
                "'Not enough information in provided documents.'\n"
            )

        format_instruction = ""
        if self.bullet_points:
            format_instruction = "Summarize the answer in 5 bullet points.\n"

        return f"""
                You are a strict financial AI assistant.

                {reasoning_instruction}
                Use ONLY the provided context to answer the question.
                {citation_instruction}
                {strict_instruction}
                {format_instruction}

                Context:
                {context}

                Question:
                {question}

                Answer:
                """


    def build_reflection_prompt(
        self,
        question: str,
        answer: str,
        context: str
    ) -> str:

            return f"""
                    You are a strict answer validator.

                    Question:
                    {question}  

                    Context:
                    {context}

                    Answer:
                    {answer}

                    Is every statement in the answer fully supported by the context?

                    Reply only with YES or NO.
                    """
    def build_rag_prompt(self,context: str, question: str) -> str:
            return f"""
        You are a helpful financial AI assistant.

        Use ONLY the context below to answer the question.
        Summarize in 5 bullets points.
        If the answer is not present in the context, say:
        "I don't have enough information to answer this."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    
    def build_prompt(self, query: str, context: str) -> str:
        return f"""
            You are a domain expert assistant.

            Use ONLY the provided context to answer the question.

            Context:
            {context}

            Question:
            {query}

            If the answer is not found in context, say "Not found in provided documents."
            """
    
    def refine_query_for_iterative_retrieval(self, query: str, answer: str) -> str:

            refinement_prompt = f"""
                                    You are a query refinement assistant.

                                    The original query:
                                    {query}

                                    The previous answer was insufficient or unsupported:
                                    {answer}

                                    Rewrite the query to improve document retrieval.
                                    Make it more specific and retrieval-focused.

                                    Refined Query:
                                    """
            return refinement_prompt

    
    def build_confidence_prompt(self, question, answer, context):

        return f"""
                You are a strict answer validator.

                Question:
                {question}

                Context:
                {context}

                Answer:
                {answer}

                Rate how well the answer is supported by the context
                on a scale from 0.0 to 1.0.

                Return only a decimal number.
                """