from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseVectorStore(ABC):

    # -------- INGESTION --------

    @abstractmethod
    def upsert(self, vectors: List[Dict[str, Any]], namespace: str = None):
        """
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
    def delete(self, ids: List[str]):
        pass

    @abstractmethod
    def update(self, id: str, vector=None, metadata=None):
        pass


    # -------- RETRIEVAL --------

    @abstractmethod
    async def dense_search(self, query_vector, top_k: int, namespace: str = None):
        pass

    @abstractmethod
    async def hybrid_search(self, query_text: str, top_k: int, alpha: float):
        pass

    @abstractmethod
    async def search(self, query_vector, top_k):
        raise NotImplementedError