from app.bootstrap import bootstrap
bootstrap()
import uuid
from typing import List, Dict, Any

from tqdm import tqdm

from app.evaluation.dataset_builder import DatasetBuilder, EvaluationSample
from app.graph.agent_graph_v2 import build_agent_graph_v2
from app.agents.state import AgentState
from scripts.test_rag_pipeline import build_rag_service
from app.mcp.client.mcp_singleton import mcp_client
import logging

logger = logging.getLogger(__name__)



class OfflineEvaluationRunner:

    def __init__(self):
        self.dataset_builder = DatasetBuilder()
        self.graph = None
        self.rag_service = None

    async def initialize(self):
        """
        Initialize services exactly like production agent run.
        """

        self.rag_service = build_rag_service()

        self.graph = await build_agent_graph_v2(self.rag_service)

    async def run(self) -> List[Dict[str, Any]]:
        """
        Run evaluation dataset through agent graph.
        """

        logger.info("Running Offline Runner")

        if self.graph is None:
            await self.initialize()

        dataset: List[EvaluationSample] = self.dataset_builder.load()

        results = []

        for sample in tqdm(dataset, desc="Running evaluation"):

            try:

                state = AgentState(
                    session_id=str(uuid.uuid4()),
                    query=sample.question
                )

                agent_result = await self.graph.ainvoke(state)

                result = {
                    "id": sample.id,
                    "question": sample.question,
                    "ground_truth": sample.ground_truth,
                    "expected_tools": sample.expected_tools,
                    "difficulty": sample.difficulty,
                    "category": sample.category,

                    "answer": agent_result.get("answer"),

                    "retrieved_docs": agent_result.get(
                        "retrieved_docs", []
                    ),

                    "tool_results": agent_result.get(
                        "tool_results", []
                    ),

                    "validation_score": agent_result.get(
                        "validation_score"
                    ),

                    "retry_count": agent_result.get(
                        "retry_count", 0
                    ),
                }

                results.append(result)

            except Exception as e:

                logger.error(f"Exception thrown while running runner {e}")

                results.append(
                    {
                        "id": sample.id,
                        "question": sample.question,
                        "error": str(e)
                    }
                )

        return results

    async def cleanup(self):
        """
        Clean up resources after evaluation.
        """

        logger.info("Cleanup")

        if hasattr(self.rag_service, "vector_store"):
            client = getattr(self.rag_service.vector_store, "client", None)

            if client:
                client.close()

        await mcp_client.close()