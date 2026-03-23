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
from app.query.expansion.llm_query_expander import LLMQueryExpander
from app.rerankers.cross_encoder_reranker import CrossEncoderReranker

from app.prompts.rag_prompt import RAGPromptBuilder
from app.pipelines.ingestion_pipeline import IngestionPipeline
from app.services.rag_service import RAGService
from app.pipelines.evaluation_pipeline import EvaluationPipeline
from app.agents.agent_state import AgentState


from app.config import OPENAI_API_KEY

# --- Agent ---
from app.graph.agent_graph import build_agent_graph
from app.mcp.client.mcp_singleton import mcp_client




async def create_application() -> Application:

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
    query_expander = LLMQueryExpander(llm)


    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=16,
    )

    retriever = RetrieverPipeline(
        base_retriever=dense_retriever,
        reranker=reranker,
        query_transformer=query_rewriter,
        query_expander=query_expander,
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

    # NOTE:
    # AgentWrapper is responsible for cleaning up shared resources
    # like vector_store client and MCP client.
    class AgentWrapper:
        def __init__(self, graph, vector_store):
            self.graph = graph
            self.vector_store = vector_store

        async def run(self, query: str, session_id: str):
            state = AgentState(
                session_id=session_id,
                query=query
            )
            return await self.graph.ainvoke(state)

        async def cleanup(self):
            client = getattr(self.vector_store, "client", None)
            if client:
                client.close()
                logger.info("Vector DB client closed")

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

    

    evaluator = EvaluationPipeline(rag_service)

    # =========================================================
    # 6. APPLICATION
    # =========================================================

    return Application(
        rag_pipeline=rag_pipeline,
        ingestion_pipeline=ingestion_pipeline,
        agent=agent,
        evaluator=evaluator
    )