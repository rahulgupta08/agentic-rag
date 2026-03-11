
# This schema represents a chunk before embedding.
# Produced by:chunker.py
# Consumed by:embedder.py
# Responsibilities
# standardize chunk structure
# store chunk metadata
# generate deterministic IDs


# DocumentChunk(
#     id="apple_2024_chunk_042",
#     text="Apple's net sales increased to $383 billion...",
#     metadata={
#         "document_id": "apple_10k_2024",
#         "collection": "financial_documents",
#         "company": "Apple",
#         "year": "2024",
#         "page": "42",
#         "source": "apple_10k_2024.pdf"
#     }
# )

# It supports:

# domain independence
# metadata enrichment
# future filtering
# evaluation debugging

from pydantic import BaseModel
from typing import Optional, Dict

class DocumentChunkSchema(BaseModel):
    id: str  # Deterministic ID based on content and metadata
    text: str
    metadata: Dict[str, Optional[str]]