from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Synchronous retrieve method for RAG mode"""
        pass

    @abstractmethod
    async def aretrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Asynchronous retrieve method for Agent mode"""
        pass

    # ALL retrievers must return:
    #     {
    #     "text": str,
    #     "score": float,   # unified key
    #     "metadata": dict  # optional
    # }