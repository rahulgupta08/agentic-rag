from app.bootstrap import bootstrap
bootstrap()
import logging
logger = logging.getLogger(__name__)
from app.factory import create_application
from app.schemas.pipeline_inputs import IngestionInput
import asyncio


async def main():
    app = await create_application()

    try:
        input = IngestionInput(
            file_path="data/raw_documents/aapl-10K_2024.pdf"
        )

        result = await app.ingest(input)
        print(result)

    finally:
        await app.cleanup()


if __name__ == "__main__":
    asyncio.run(main())