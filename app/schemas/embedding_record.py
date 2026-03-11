# This represents a vectorized chunk ready for storage.

# Produced by: embedder.py
# Consumed by:
# vectorstores/

# Responsibilities
# store embedding vector
# carry chunk text
# carry metadata
# enable DB adapters

# EmbeddingRecordSchema(
#     id="apple_2024_chunk_042",
#     vector=[0.12, -0.31, 0.45, ...],
#     text="Apple's net sales increased to $383 billion...",
#     metadata={
#         "document_id": "apple_10k_2024",
#         "collection": "financial_documents",
#         "company": "Apple",
#         "year": "2024"
#     }
# )

# Why This Schema Matters

# It allows your adapters to convert into:

# Pinecone vector format
# Weaviate object format
# future vector DBs

# without changing ingestion logic.

from pydantic import BaseModel
from typing import List, Dict, Optional


class EmbeddingRecordSchema(BaseModel):

    id: str

    vector: List[float]

    text: str

    metadata: Dict[str, Optional[str]]