from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        pass

        # ALL retrievers must return:
        #     {
        #     "text": str,
        #     "score": float,   # unified key
        #     "metadata": dict  # optional
        # }