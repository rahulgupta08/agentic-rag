from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseVectorStore(ABC):

    # -------- INGESTION --------

    @abstractmethod
    def upsert(self, vectors: List[Dict[str, Any]], namespace: str = None):
        """Synchronous upsert for RAG mode
        vectors = [
            {
                "id": str,
                "vector": List[float],
                "text": str,
                "metadata": dict
            }
        ]
        """
        pass

    @abstractmethod
    async def aupsert(self, vectors: List[Dict[str, Any]], namespace: str = None):
        """Asynchronous upsert for Agent mode"""
        pass

    @abstractmethod
    def delete(self, ids: List[str]):
        """Synchronous delete for RAG mode"""
        pass

    @abstractmethod
    async def adelete(self, ids: List[str]):
        """Asynchronous delete for Agent mode"""
        pass

    @abstractmethod
    def update(self, id: str, vector=None, metadata=None):
        """Synchronous update for RAG mode"""
        pass

    @abstractmethod
    async def aupdate(self, id: str, vector=None, metadata=None):
        """Asynchronous update for Agent mode"""
        pass

    # -------- RETRIEVAL --------

    @abstractmethod
    def dense_search(self, query_vector, top_k: int, namespace: str = None):
        """Synchronous dense search for RAG mode"""
        pass

    @abstractmethod
    async def dense_search_async(self, query_vector, top_k: int, namespace: str = None):
        """Asynchronous dense search for Agent mode"""
        pass

    @abstractmethod
    def hybrid_search(self, query_text: str, top_k: int, alpha: float):
        """Synchronous hybrid search for RAG mode"""
        pass

    @abstractmethod
    async def hybrid_search_async(self, query_text: str, top_k: int, alpha: float):
        """Asynchronous hybrid search for Agent mode"""
        pass

    @abstractmethod
    def search(self, query_vector, top_k):
        """Synchronous search for RAG mode"""
        raise NotImplementedError

    @abstractmethod
    async def search_async(self, query_vector, top_k):
        """Asynchronous search for Agent mode"""
        raise NotImplementedError