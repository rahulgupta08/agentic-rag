from app.bootstrap import bootstrap
bootstrap()

import logging
logger = logging.getLogger(__name__)

import asyncio
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from app.rag.rag_pipeline import RAGPipeline
from app.services.rag_service import RAGService

from app.vectorstores.factory import get_vector_store
from app.config import OPENAI_API_KEY

from app.embeddings.local_embedder import LocalEmbedder

# ✅ Current retriever (ACTIVE)
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.retriever_pipeline import RetrieverPipeline

# 🔥 Future-ready imports (NOT USED YET)
# from app.retrieval.bm25_retriever import BM25Retriever
# from app.retrieval.hybrid_retriever import HybridRetriever
# from app.retrieval.fusion_strategy import WeightedFusion

from app.rerankers.cross_encoder_reranker import CrossEncoderReranker
from app.prompts.rag_prompt import RAGPromptBuilder

load_dotenv()


def build_rag_service():

    # -----------------------------
    # Vector Store + Embedder
    # -----------------------------
    vector_store = get_vector_store()
    embedder = LocalEmbedder()

    # -----------------------------
    # Dense Retriever (ACTIVE)
    # -----------------------------
    base_retriever = DenseRetriever(vector_store, embedder)

    # -----------------------------
    # Reranker
    # -----------------------------
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=16,
    )

    # -----------------------------
    # Retriever Pipeline
    # -----------------------------
    retriever = RetrieverPipeline(
        base_retriever=base_retriever,
        reranker=reranker,
        # post_processors=[]  # keep optional
        logger=logger
    )

    # -----------------------------
    # LLM
    # -----------------------------
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0
    )

    # -----------------------------
    # Prompt Builder
    # -----------------------------
    prompt_builder = RAGPromptBuilder()

    # -----------------------------
    # RAG Service
    # -----------------------------
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
    logger.info(f"\nDOCUMENTS USED: {len(response['documents'])}")


if __name__ == "__main__":
    asyncio.run(main())