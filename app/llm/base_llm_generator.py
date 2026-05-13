from abc import ABC, abstractmethod

class BaseLLMGenerator(ABC):
    
    @abstractmethod
    def generate(self, query: str, context: str) -> str:
     pass