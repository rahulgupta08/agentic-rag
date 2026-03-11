import asyncio
from openai import OpenAI

from app.rag.rag_pipeline import RAGPipeline
from app.services.rag_service import RAGService
from app.retrieval.dense_retriever import DenseRetriever
from app.vectorstores.factory import get_vector_store
from app.prompts.rag_prompt import RAGPromptBuilder
from app.config import OPENAI_API_KEY


def build_rag_service():

    # Vector store
    vector_store = get_vector_store()

    # Retriever
    retriever = DenseRetriever(
        vector_store=vector_store,
        embedder=None,      # assuming embedder handled elsewhere
        reranker=None
    )

    # LLM
    llm = OpenAI(api_key=OPENAI_API_KEY)

    # Prompt builder
    prompt_builder = RAGPromptBuilder()

    # RAG service
    rag_service = RAGService(
        retriever=retriever,
        llm=llm,
        prompt_builder=prompt_builder
    )

    return rag_service


async def main():

    query = "What risk factors does Apple mention in the 10-K?"

    print("\nQUERY:")
    print(query)

    rag_service = build_rag_service()

    pipeline = RAGPipeline(rag_service)

    response = await pipeline.run(query)

    print("\nANSWER:\n")
    print(response["answer"])

    print("\nDOCUMENTS USED:", len(response["documents_used"]))


if __name__ == "__main__":
    asyncio.run(main())