from app.factory import create_application
import asyncio


async def main():
    app = await create_application()

    try:
        result = await app.evaluate()
        print(result)

    finally:
        await app.cleanup()


if __name__ == "__main__":
    asyncio.run(main())