from abc import ABC, abstractmethod
from typing import List, Dict


class BaseReranker(ABC):

    @abstractmethod
    def rerank(self, query: str, documents: List, top_k: int = 5) -> List:
        """Synchronous rerank for RAG mode"""
        pass

    @abstractmethod
    async def arerank(self, query: str, documents: List, top_k: int = 5) -> List:
        """Asynchronous rerank for Agent mode"""
        pass