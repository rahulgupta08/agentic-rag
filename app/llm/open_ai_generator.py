import logging
logger = logging.getLogger(__name__)

from app.llm.base_llm_generator import BaseLLMGenerator

from app.config import OPENAI_API_KEY
from app.core.exceptions import LLMGenerationError

try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    raise LLMGenerationError(f"Failed to initialize OpenAI client: {str(e)}")


class OpenAIGenerator(BaseLLMGenerator):
    def __init__(self, model: str = "gpt-4-mini", api_key: str = None):
        self.model = model
        self.client = OpenAI(api_key=api_key or OPENAI_API_KEY)

    def generate(self, query: str, context: str) -> str:
        try:
            prompt = f"""
                        You are a helpful AI assistant.

                        Answer the question strictly using the provided context.
                        If the answer is not in the context, say "I don't know based on the provided information."

                        Context:
                        {context}

                        Question:
                        {query}
                        """

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            answer = response.choices[0].message.content

            logger.info("LLM response generated successfully")
            return answer

        except Exception as e:
            logger.exception("LLM generation failed")
            raise LLMGenerationError(str(e))