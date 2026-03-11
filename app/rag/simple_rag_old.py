class SimpleRAG:

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def ask(self, query):
        docs = self.retriever.retrieve(query)

        context = "\n\n".join([d["text"] for d in docs])

        prompt = f"""
        Use only the context below to answer.

        Context:
        {context}

        Question:
        {query}
        """

        return self.llm.generate(prompt)