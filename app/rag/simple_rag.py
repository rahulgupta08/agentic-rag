from app.bootstrap import bootstrap
bootstrap()
import logging
logger = logging.getLogger(__name__)
from openai import OpenAI
from app.vectorstores.factory import get_vector_store
from app.ingestion.embedder import embed_text
from app.prompts.rag_prompt import RAGPromptBuilder
from app.config import OPENAI_API_KEY


MAX_CONTEXT_CHARS = 6000

class SimpleRAG:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.llm = OpenAI(api_key=OPENAI_API_KEY)
        self.prompt_builder = RAGPromptBuilder()

    def retrieve(self, query: str, top_k: int = 4):
        query_vector = embed_text(query)
        
        logger.info(f"Query  {query}")
        logger.info(f"Query vector {len(query_vector)}")

        #results = self.vector_store.query(
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k
        )

        return results

    def generate(self, context: str, question: str):
        
        prompt = self.prompt_builder.build_generation_prompt(context, question)

        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        logger.info(f"LLM response:  {response}")

        return response.choices[0].message.content

    def ask(self, question: str, top_k: int = 4):

        logger.info(f"Asking question:  {question}")

        docs = self.retrieve(question, top_k=top_k)

        
        
        logger.info(f"Retrieved {len(docs)} documents")

        if not docs:
            return {
                "question": question,
                "answer": "No relevant documents found.",
                "retrieved_docs": []
            }
        else:
            logger.info("Retrieved Documents:")

            for d in docs:
                logger.info(f" ID : {d.get("id")}, Score: {d.get("score")} ,Preview: {d.get("text", "")[:200]} {'-'* 60}")

        context = ""

        for doc in docs:
            if len(context) + len(doc["text"]) > MAX_CONTEXT_CHARS:
                break
            context += doc["text"] + "\n\n"
        
        logger.info(f"Context for generation: {context}")

        answer = self.generate(context=context, question=question)

        return {
            "question": question,
            "answer": answer,
            "retrieved_docs": docs
        }