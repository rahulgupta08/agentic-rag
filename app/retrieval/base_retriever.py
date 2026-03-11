from abc import ABC, abstractmethod

class BaseRetriever(ABC):

    @abstractmethod
    async def asyncretrieve(self, query: str, top_k: int = 5):
        pass