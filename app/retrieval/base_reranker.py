from abc import ABC, abstractmethod
from typing import List, Dict


class BaseReranker(ABC):

    @abstractmethod
    async def rerank(self,
        query: str,
        documents: List[Dict],
        top_k: int
    ) -> List[Dict]:
        pass