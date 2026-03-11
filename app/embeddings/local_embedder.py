from typing import List
from sentence_transformers import SentenceTransformer

from .base_embedder import BaseEmbedder


class LocalEmbedder(BaseEmbedder):

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):

        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

    @property
    def dimension(self):
        return self._dimension

    

    def embed_query(self, text: str) -> List[float]:

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return embeddings.tolist()