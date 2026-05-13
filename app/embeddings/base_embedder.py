from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Synchronous query embedding for RAG mode"""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Synchronous batch embedding for RAG mode"""
        pass

    @abstractmethod
    async def embed_query_async(self, text: str) -> List[float]:
        """Asynchronous query embedding for Agent mode"""
        pass

    @abstractmethod
    async def embed_batch_async(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous batch embedding for Agent mode"""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embeddings"""
        pass