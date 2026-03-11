from typing import List
from openai import OpenAI
from .base_embedder import BaseEmbedder
from dotenv import load_dotenv

load_dotenv()


class OpenAIEmbedder(BaseEmbedder):

    def __init__(self, model: str):
        self.client = OpenAI()
        self.model = model
        if model == "text-embedding-3-large":
            self._dimension = 3072
        elif model == "text-embedding-3-small":
            self._dimension = 1536
        else:
            raise ValueError("Unknown OpenAI embedding model")

    @property
    def dimension(self):
        return self._dimension


    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )

        return response.data[0].embedding


    def embed_batch(self, texts: List[str]) -> List[List[float]]:

        response = self.client.embeddings.create(
            model=self.model,
            input=texts   # OpenAI supports batch input
        )

        return [item.embedding for item in response.data]