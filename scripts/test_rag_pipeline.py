from app.bootstrap import bootstrap
bootstrap()
import logging
logger = logging.getLogger(__name__)
from app.factory import create_application
from app.schemas.pipeline_inputs import RAGInput


import asyncio

async def main():
    app = await create_application()

    try:
        input = RAGInput(
            query="What risk factors does Apple mention in the 10-K?"
        )

        response = await app.run_rag(input)
        print(response)

    finally:
        await app.cleanup()

    logger.info(f"Respnse : {response}")

if __name__ == "__main__":
    asyncio.run(main())