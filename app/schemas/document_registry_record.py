# This represents a document entry in the SQLite registry.

# Produced by:

# ingestion_pipeline.py

# Stored in:

# registry.db
# Responsibilities
# track document lifecycle
# detect duplicate ingestion
# detect document updates
# track embedding configuration

# DocumentRegistryRecordSchema(
#     document_id="apple_10k_2024",
#     collection="financial_documents",
#     source_path="data/raw_documents/apple_10k_2024.pdf",
#     checksum="f4a2c98c...",
#     chunk_count=438,
#     embedding_model="text-embedding-3-large",
#     chunk_size=800,
#     ingested_at=datetime.now()
# )



from pydantic import BaseModel
from datetime import datetime


class DocumentRegistryRecordSchema(BaseModel):

    document_id: str

    collection: str

    source_path: str

    checksum: str

    chunk_count: int

    embedding_model: str

    chunk_size: int

    ingested_at: datetime

    ingestion_config_hash: str