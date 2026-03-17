
from app.schemas.pipeline_inputs import (
    RAGInput,
    AgentInput,
    IngestionInput
)
from app.agents.agent_state import AgentState

class Application:

    def __init__(
        self,
        rag_pipeline,
        ingestion_pipeline,
        agent,
        evaluator
    ):
        self.rag_pipeline = rag_pipeline
        self.ingestion_pipeline = ingestion_pipeline
        self.agent = agent
        self.evaluator = evaluator

    async def run_rag(self, rag_input: RAGInput):
        return await self.rag_pipeline.run(rag_input.query)

    async def run_agent(self, agent_input: AgentInput):
        return await self.agent.run(
            query=agent_input.query,
            session_id=agent_input.session_id
        )

    async def ingest(self, ingestion_input: IngestionInput):
        return await self.ingestion_pipeline.run(
            file_path=ingestion_input.file_path,
            collection=ingestion_input.collection
        )

    async def evaluate(self):
        return await self.evaluator.run()

    async def cleanup(self):

        if hasattr(self.ingestion_pipeline, "cleanup"):
            await self.ingestion_pipeline.cleanup()

        if hasattr(self.agent, "cleanup"):
            await self.agent.cleanup()

        if hasattr(self.evaluator, "cleanup"):
            await self.evaluator.cleanup()