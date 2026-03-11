import hashlib
import json
from pathlib import Path
from app.schemas.embedding_record import EmbeddingRecordSchema
from typing import List, Dict
import uuid


def generate_ingestion_config_hash(config: dict) -> str:
    """
    Generate a deterministic hash representing the ingestion configuration.
    Used to detect ingestion pipeline changes that require re-indexing.
    """

    config_str = json.dumps(config, sort_keys=True)

    return hashlib.md5(config_str.encode()).hexdigest()

def compute_document_checksum(file_path: str) -> str:
    """
    Compute SHA256 checksum of a document.
    Used to detect document changes.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()

def generate_document_id(file_path: str) -> str:
    """
    Generate document_id from file name.
    """

    filename = Path(file_path).stem

    return filename.lower()

def embedding_records_to_vectors(
        records: List[EmbeddingRecordSchema]
    ) -> List[Dict]:

        vectors = []

        for record in records:
            vectors.append({
                "id": record.id,
                "vector": record.vector,
                "text": record.text,
                "metadata": {
                    **record.metadata,
                    "text": record.text
                }
            })


        return vectors

def generate_deterministic_uuid(value: str) -> str:
    """
    Generate deterministic UUID from a string.
    Needed for Weaviate which requires UUID format.
    """

    return str(uuid.uuid5(uuid.NAMESPACE_DNS, value))

def generate_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()