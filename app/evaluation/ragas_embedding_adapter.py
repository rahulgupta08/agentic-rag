import asyncio
from typing import List
from ragas.embeddings import BaseRagasEmbeddings



class RagasEmbeddingAdapter(BaseRagasEmbeddings):
    """
    Adapter to use LocalEmbedder with RAGAS.
    """

    def __init__(self, embedder):
        self.embedder = embedder

    def embed_query(self, text: str) -> List[float]:
        return  self.embedder.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return  self.embedder.embed_batch(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        return await self.embedder.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self.embedder.embed_batch(texts)