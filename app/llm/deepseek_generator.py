from app.llm.base_llm_generator import BaseLLMGenerator
from app.core.exceptions import LLMGenerationError
from app.config import OPENROUTER_API_KEY

import logging
import asyncio
from functools import partial
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except Exception as e:
    raise LLMGenerationError(f"Failed to import OpenAI client: {str(e)}")


class DeepSeekGenerator(BaseLLMGenerator):
    def __init__(self, model: str = "deepseek/deepseek-v3.2", api_key: str = None):
        try:
            self.model = model
            self.client = OpenAI(
                api_key=api_key or OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1"
            )
        except Exception as e:
            raise LLMGenerationError(f"Failed to initialize DeepSeek client: {str(e)}")

    def invoke(self, prompt: str) -> str:
        logger.info(f"Synchronous invoke method for RAG mode")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise and factual assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,   # lower = better for RAG
                max_tokens=512
            )

            answer = response.choices[0].message.content
            logger.info("DeepSeek V3.2 response generated successfully")
            return answer.strip()

        except Exception as e:
            logger.exception("DeepSeek generation failed")
            raise LLMGenerationError(str(e))

    def generate(self, query: str, context: str) -> str:
        """Synchronous generate method for RAG mode"""
        prompt = self._build_prompt(query, context)
        return self.invoke(prompt)

    async def ainvoke(self, prompt: str) -> str:
        
        logger.info(f"Asynchronous ainvoke method for Agent mode")
        try:
            # Run the synchronous invoke in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                self.invoke,
                prompt
            )
            return response
        except Exception as e:
            logger.exception("DeepSeek async generation failed")
            raise LLMGenerationError(str(e))

    async def agenerate(self, query: str, context: str) -> str:
        """Asynchronous generate method for Agent mode"""
        prompt = self._build_prompt(query, context)
        return await self.ainvoke(prompt)

    def _build_prompt(self, query: str, context: str) -> str:
        """Helper method to build the prompt"""
        return f"""
            You are a helpful AI assistant.

            STRICT RULES:
            - Answer ONLY using the provided context
            - Do NOT hallucinate
            - If answer is missing → say "I don't know based on the provided information"

            Context:
            {context}

            Question:
            {query}
            """