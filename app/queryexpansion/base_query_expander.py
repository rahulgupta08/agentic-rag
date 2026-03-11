from abc import ABC, abstractmethod


class BaseQueryExpander(ABC):

    @abstractmethod
    def expand(self, query: str) -> str:
        pass