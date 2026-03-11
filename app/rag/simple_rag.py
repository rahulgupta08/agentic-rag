from openai import OpenAI
from app.vectorstores.factory import get_vector_store
from app.ingestion.embedder import embed_text
from app.prompts.rag_prompt import RAGPromptBuilder
from app.config import OPENAI_API_KEY
import logging

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 6000

class SimpleRAG:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.llm = OpenAI(api_key=OPENAI_API_KEY)
        self.prompt_builder = RAGPromptBuilder()

    def retrieve(self, query: str, top_k: int = 4):
        query_vector = embed_text(query)
        
        print("Query " , query)
        print("Query vector " , len(query_vector))

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

        print("LLM response: " , response)

        return response.choices[0].message.content

    def ask(self, question: str, top_k: int = 4):

        print("Asking question: " , question)

        docs = self.retrieve(question, top_k=top_k)

        
        print(f"No of docs retrieved" , len(docs))
        logger.info(f"Retrieved {len(docs)} documents")

        if not docs:
            return {
                "question": question,
                "answer": "No relevant documents found.",
                "retrieved_docs": []
            }
        else:
            print("\nRetrieved Documents:\n")

            for d in docs:
                print("ID:", d.get("id"))
                print("Score:", d.get("score"))
                print("Preview:", d.get("text", "")[:200])
                print("-" * 60)

        context = ""

        for doc in docs:
            if len(context) + len(doc["text"]) > MAX_CONTEXT_CHARS:
                break
            context += doc["text"] + "\n\n"
        
        print("Context for generation: " , context)

        answer = self.generate(context=context, question=question)

        return {
            "question": question,
            "answer": answer,
            "retrieved_docs": docs
        }