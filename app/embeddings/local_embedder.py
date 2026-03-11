from typing import List
from sentence_transformers import SentenceTransformer
import asyncio

from .base_embedder import BaseEmbedder


class LocalEmbedder(BaseEmbedder):

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):

        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

    @property
    def dimension(self):
        return self._dimension

    

    async def embed_query(self, text: str) -> List[float]:

        embedding = await asyncio.to_thread(
            self.model.encode,
            text,
            normalize_embeddings=True
        )

        return embedding.tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:

        embeddings = await asyncio.to_thread(
            self.model.encode,
            texts,
            normalize_embeddings=True
        )

        return embeddings.tolist()