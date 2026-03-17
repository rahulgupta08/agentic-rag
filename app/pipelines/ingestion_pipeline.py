from app.bootstrap import bootstrap
bootstrap()
import logging
logger = logging.getLogger(__name__)
from typing import Dict, Any

from app.ingestion.document_loader import DocumentLoader
from app.ingestion.chunker_new import DocumentChunker
from app.ingestion.ingestion_utils import (
    generate_document_id,
    embedding_records_to_vectors
)
from app.schemas.embedding_record import EmbeddingRecordSchema
from app.ingestion.metadata_extractor import extract_metadata_from_filename


class IngestionPipeline:

    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    async def run(self, file_path: str, collection: str = "financial_documents") -> Dict[str, Any]:

        try:
            logger.info(f"Starting ingestion for file: {file_path}")

            # -----------------------------
            # Load Document
            # -----------------------------
            loader = DocumentLoader(file_path)
            document = loader.load()

            logger.info(f"File loaded: {file_path}")

            # -----------------------------
            # Metadata Extraction
            # -----------------------------
            document_metadata = extract_metadata_from_filename(file_path)
            logger.info(f"Extracted metadata: {document_metadata}")

            # -----------------------------
            # Generate Document ID
            # -----------------------------
            document_id = generate_document_id(file_path)

            # -----------------------------
            # Chunking
            # -----------------------------
            chunker = DocumentChunker()

            chunks = chunker.chunk(
                document=document,
                document_id=document_id,
                collection=collection,
                document_metadata=document_metadata
            )

            logger.info(f"Total chunks created: {len(chunks)}")

            # -----------------------------
            # Embedding
            # -----------------------------
            texts = [chunk.text for chunk in chunks]

            embeddings = await self.embedder.embed_batch(texts)

            logger.info(f"Embeddings generated: {len(embeddings)}")

            # -----------------------------
            # Build Embedding Records
            # -----------------------------
            embedding_records = []

            for chunk, vector in zip(chunks, embeddings):
                record = EmbeddingRecordSchema(
                    id=chunk.id,
                    vector=vector,
                    text=chunk.text,
                    metadata=chunk.metadata
                )
                embedding_records.append(record)

            logger.info(f"Embedding records created: {len(embedding_records)}")

            # -----------------------------
            # Convert for Vector DB
            # -----------------------------
            vectors_for_db = embedding_records_to_vectors(embedding_records)

            logger.info(f"Vectors to upsert: {len(vectors_for_db)}")

            # -----------------------------
            # Upsert to Vector Store
            # -----------------------------
            self.vector_store.upsert(
                vectors=vectors_for_db,
                namespace=collection
            )

            logger.info(f"Inserted vectors: {len(vectors_for_db)}")

            # -----------------------------
            # Return Structured Output
            # -----------------------------
            return {
                "file_path": file_path,
                "collection": collection,
                "num_chunks": len(chunks),
                "num_embeddings": len(embeddings),
                "num_vectors_inserted": len(vectors_for_db),
                "metadata": document_metadata
            }

        except Exception:
            logger.exception("Ingestion pipeline failed")
            raise

    async def cleanup(self):
        """
        Optional cleanup (vector DB client, if present)
        """
        client = getattr(self.vector_store, "client", None)
        if client:
            client.close()
            logger.info("Vector DB client closed")