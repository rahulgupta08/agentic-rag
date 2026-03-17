from app.factory import create_application
from app.schemas.pipeline_inputs import AgentInput
import asyncio
import uuid


async def main():
    app = await create_application()

    try:
        input = AgentInput(
            query="What is Apple's latest stock price?",
            session_id=str(uuid.uuid4())
        )

        result = await app.run_agent(input)
        print(result)

    finally:
        await app.cleanup()


if __name__ == "__main__":
    asyncio.run(main())