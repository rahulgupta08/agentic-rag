
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

    async def run_rag(self, input: RAGInput):
        return await self.rag_pipeline.run(input.query)

    async def run_agent(self, input: AgentInput):
        state = AgentState(
            session_id=input.session_id,
            query=input.query
        )
        return await self.agent.run(state)

    async def ingest(self, input: IngestionInput):
        return await self.ingestion_pipeline.run(
            file_path=input.file_path,
            collection=input.collection
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