from app.bootstrap import bootstrap
bootstrap()

import logging
logger = logging.getLogger(__name__)

import asyncio
import uuid

from langchain_openai import ChatOpenAI

from app.graph.agent_graph import build_agent_graph
from app.agents.agent_state import AgentState

from app.vectorstores.factory import get_vector_store
from app.embeddings.local_embedder import LocalEmbedder

from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.retriever_pipeline import RetrieverPipeline

from app.query.rewriting.llm_query_rewriter import LLMQueryRewriter

from app.rerankers.cross_encoder_reranker import CrossEncoderReranker
from app.prompts.rag_prompt import RAGPromptBuilder
from app.services.rag_service import RAGService
from app.config import OPENAI_API_KEY

from app.mcp.client.mcp_singleton import mcp_client


async def run_agent():

    logger.info(" Running Graph Agent")

    # -----------------------------
    # LLM (shared across components)
    # -----------------------------
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0
    )

    # -----------------------------
    # Vector Store + Embedder
    # -----------------------------
    vector_store = get_vector_store()
    embedder = LocalEmbedder()

    # -----------------------------
    # Retriever Components
    # -----------------------------
    dense_retriever = DenseRetriever(vector_store, embedder)

    query_rewriter = LLMQueryRewriter(llm=llm)

    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=16,
    )

    # -----------------------------
    # Retriever Pipeline
    # -----------------------------
    retriever = RetrieverPipeline(
        base_retriever=dense_retriever,
        reranker=reranker,
        query_transformer=query_rewriter,
        logger=logger
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

    # -----------------------------
    # Build Agent Graph
    # -----------------------------
    graph = await build_agent_graph(rag_service)

    # -----------------------------
    # Initial State
    # -----------------------------
    query = "What is Apple's latest stock price?"

    state = AgentState(
        session_id=str(uuid.uuid4()),
        query=query
    )

    logger.info(f" USER QUERY: {query}")

    try:
        # -----------------------------
        # Run Agent
        # -----------------------------
        result = await graph.ainvoke(state)

        logger.info("===== FINAL ANSWER =====")
        logger.info(result.get("answer"))

    except Exception:
        logger.exception(" Agent execution failed")
        raise

    finally:
        # -----------------------------
        # Cleanup
        # -----------------------------
        client = getattr(vector_store, "client", None)
        if client:
            client.close()
            logger.info(" Vector DB client closed")

        await mcp_client.close()
        logger.info(" MCP client closed")


if __name__ == "__main__":
    asyncio.run(run_agent())