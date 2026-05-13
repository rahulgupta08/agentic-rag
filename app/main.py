from app.bootstrap import bootstrap
bootstrap()
import logging
logger = logging.getLogger(__name__)
import asyncio
import argparse
import uuid

from app.factory import create_application
from app.schemas.pipeline_inputs import (
    RAGInput,
    AgentInput,
    IngestionInput
)


async def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", required=True, choices=[
        "rag",
        "agent",
        "ingest",
        "eval"
    ])

    parser.add_argument("--query", type=str)
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--collection", type=str, default="financial_documents")
    parser.add_argument("--session_id", type=str)

    args = parser.parse_args()

    app = await create_application()

    try:

        # -----------------------------
        # Validation
        # -----------------------------
        if args.mode in ["rag", "agent"] and not args.query:
            raise ValueError("--query is required for rag and agent modes")

        if args.mode == "ingest" and not args.file_path:
            raise ValueError("--file_path is required for ingest mode")

        # -----------------------------
        # Execution
        # -----------------------------
        if args.mode == "rag":
            input = RAGInput(query=args.query)
            # Use synchronous run() for RAG mode
            result = app.rag_pipeline.run(input.query)

        elif args.mode == "agent":
            session_id = args.session_id or str(uuid.uuid4())

            input = AgentInput(
                query=args.query,
                session_id=session_id
            )
            # Use asynchronous run() for Agent mode
            result = await app.run_agent(input)

        elif args.mode == "ingest":
            input = IngestionInput(
                file_path=args.file_path,
                collection=args.collection
            )
            result = await app.ingest(input)

        elif args.mode == "eval":
            result = await app.evaluate()

        logger.info(f"Results retrieved : {result}")

    finally:
        await app.cleanup()


if __name__ == "__main__":
    asyncio.run(main())