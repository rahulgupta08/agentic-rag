
from app.llm.base_llm_generator import BaseLLMGenerator
from app.llm.open_ai_generator import OpenAIGenerator
from app.llm.deepseek_generator import DeepSeekGenerator
import logging
logger = logging.getLogger(__name__)




class LLMGeneratorFactory:

    @staticmethod
    def create(provider: str = "openai", model: str = None) -> BaseLLMGenerator:
            
            logger.info(f"LLM Provider : {provider}")
            
            if provider == "openai":
                return OpenAIGenerator(model=model or "gpt-4-mini")
            elif provider == "deepseek":
                return DeepSeekGenerator(model=model or "deepseek/deepseek-chat")
            else:
                raise ValueError(f"Unsupported provider: {provider}")