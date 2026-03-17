# --- Bootstrap + Logging ---
from app.bootstrap import bootstrap
bootstrap()
import logging
logger = logging.getLogger(__name__)

from app.application import Application
from app.rag.rag_pipeline import RAGPipeline



# --- Core Imports (from your repo) ---
from langchain_openai import ChatOpenAI

from app.vectorstores.factory import get_vector_store
from app.embeddings.local_embedder import LocalEmbedder

from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.retriever_pipeline import RetrieverPipeline

from app.query.rewriting.llm_query_rewriter import LLMQueryRewriter
from app.rerankers.cross_encoder_reranker import CrossEncoderReranker

from app.prompts.rag_prompt import RAGPromptBuilder
from app.pipelines.ingestion_pipeline import IngestionPipeline
from app.services.rag_service import RAGService
from app.pipelines.evaluation_pipeline import EvaluationPipeline

from app.config import OPENAI_API_KEY

# --- Agent ---
from app.graph.agent_graph import build_agent_graph

# --- Evaluation ---
from app.evaluation.offline_runner import OfflineEvaluationRunner
from app.mcp.client.mcp_singleton import mcp_client



async def create_application():

    # =========================================================
    # 1. SHARED COMPONENTS (Used across RAG + Agent)
    # =========================================================

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0
    )

    vector_store = get_vector_store()
    embedder = LocalEmbedder()

    dense_retriever = DenseRetriever(vector_store, embedder)

    query_rewriter = LLMQueryRewriter(llm=llm)

    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=16,
    )

    retriever = RetrieverPipeline(
        base_retriever=dense_retriever,
        reranker=reranker,
        query_transformer=query_rewriter,
        logger=logger
    )

    prompt_builder = RAGPromptBuilder()

    rag_service = RAGService(
        retriever=retriever,
        llm=llm,
        prompt_builder=prompt_builder
    )

    # =========================================================
    # 2. RAG PIPELINE (Already exists in your repo)
    # =========================================================

    rag_pipeline = RAGPipeline(rag_service)

    # =========================================================
    # 3. AGENT (Graph-based)
    # =========================================================

    agent_graph = await build_agent_graph(rag_service)

    # Wrap agent execution so Application can call it uniformly
    class AgentWrapper:
        async def run(self, state):
            return await agent_graph.ainvoke(state)
        
        async def cleanup(self):
            # -----------------------------
            # Vector DB cleanup
            # -----------------------------
            client = getattr(self.vector_store, "client", None)
            if client:
                client.close()
                logger.info("Vector DB client closed")

                # -----------------------------
                # MCP cleanup
                # -----------------------------
                await mcp_client.close()
            logger.info("MCP client closed")

    agent = AgentWrapper(
        graph=agent_graph,
        vector_store=vector_store
    )

    # =========================================================
    # 4. INGESTION (Currently script-based → wrap it)
    # =========================================================


    ingestion_pipeline = IngestionPipeline(
                            vector_store=vector_store,
                            embedder=embedder
                        )

    # =========================================================
    # 5. EVALUATION (Offline Runner)
    # =========================================================

    class EvaluationWrapper:
        def __init__(self):
            self.runner = OfflineEvaluationRunner()

        async def run(self, *args, **kwargs):
            return await self.runner.run()

    evaluator = EvaluationPipeline()

    # =========================================================
    # 6. APPLICATION
    # =========================================================

    return Application(
        rag_pipeline=rag_pipeline,
        ingestion_pipeline=ingestion_pipeline,
        agent=agent,
        evaluator=evaluator
    )