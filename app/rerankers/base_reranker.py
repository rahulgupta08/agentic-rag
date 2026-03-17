from abc import ABC, abstractmethod
from typing import List, Dict


class BaseReranker(ABC):

    @abstractmethod
    async def rerank(self, query, documents, top_k: int):
        pass