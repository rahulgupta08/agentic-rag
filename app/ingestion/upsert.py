
from app.bootstrap import bootstrap
bootstrap()
import logging
logger = logging.getLogger(__name__)
from pinecone import Pinecone, ServerlessSpec
from app.config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from app.core.exceptions import VectorStoreError
import os
from dotenv import load_dotenv
import uuid

load_dotenv()

pc = Pinecone(api_key=PINECONE_API_KEY)

def create_index_if_not_exists(dimension: int):
    try:
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            logger.info("Pinecone index created.")
        else:
            logger.info("Pinecone index already exists.")

    except Exception as e:
        logger.error(f"Index creation failed: {str(e)}")
        raise VectorStoreError(f"Index creation failed: {str(e)}")

def upsert_vectors(chunks: list[str], embeddings: list[list[float]], source: str):
    try:
        index = pc.Index(PINECONE_INDEX_NAME)

        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": f"{source}_chunk_{i}",
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "source": source
                }
            })

        BATCH_SIZE = 100

        for i in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[i:i + BATCH_SIZE]
            index.upsert(vectors=batch)

        logger.info(f"Upserted {len(vectors)} vectors to Pinecone.")

    except Exception as e:
        logger.error(f"Pinecone upsert failed: {str(e)}")
        raise VectorStoreError(f"Pinecone upsert failed: {str(e)}")