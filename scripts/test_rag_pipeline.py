from app.bootstrap import bootstrap
bootstrap()
import logging
logger = logging.getLogger(__name__)
import asyncio
from langchain_openai import ChatOpenAI

from app.rag.rag_pipeline import RAGPipeline
from app.services.rag_service import RAGService
from app.retrieval.dense_retriever import DenseRetriever
from app.vectorstores.factory import get_vector_store
from app.prompts.rag_prompt import RAGPromptBuilder
from app.config import OPENAI_API_KEY
from app.embeddings.local_embedder import LocalEmbedder
from app.retrieval.retriever_pipeline import RetrieverPipeline
from app.rerankers.cross_encoder_reranker import CrossEncoderReranker

from dotenv import load_dotenv

load_dotenv()


def build_rag_service():

    # Vector store
    vector_store = get_vector_store()

    embedder = LocalEmbedder()

    base_retriever = DenseRetriever(vector_store, embedder)

    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=16,
    )

    retriever = RetrieverPipeline(
        base_retriever=base_retriever,
        reranker=reranker,
    )


    # LLM
    llm = ChatOpenAI(api_key=OPENAI_API_KEY,
                     model="gpt-4o-mini",
    temperature=0)

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

    logger.info(f"\nQUERY: {query}")

    rag_service = build_rag_service()

    pipeline = RAGPipeline(rag_service)

    response = await pipeline.run(query)

    logger.info(f"\nANSWER:\n{response['answer']}")


    #logger.info(f"\nDOCUMENTS USED: {len(response['documents'])}")


if __name__ == "__main__":
    asyncio.run(main())