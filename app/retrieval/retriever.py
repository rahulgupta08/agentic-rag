# app/retrieval/retriever.py

import logging
from typing import List, Dict
from pinecone import Pinecone
from app.core.exceptions import RetrievalError
from app.retrieval.query_embedder import embed_query
from app.config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from app.vectorstores.factory import get_vector_store


logger = logging.getLogger(__name__)

class Retriever:

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.vector_store = get_vector_store()

    def retrieve(self, query: str) -> List[Dict]:
        try:
            # 1️⃣ Embed query
            query_vector = embed_query(query)

            # 2️⃣ Search Pinecone
            response = self.vector_store.query(
                vector=query_vector,
                top_k=self.top_k,
                include_metadata=True,
            )

            matches = response.get("matches", [])

            logger.info(f"Retrieved {len(matches)} documents")

            return [
                {
                    "id": match["id"],
                    "score": match["score"],
                    "text": match["metadata"].get("text", ""),
                }
                for match in matches
            ]

        except Exception as e:
            logger.exception("Retrieval failed")
            raise RetrievalError(f"Retrieval failed: {str(e)}")