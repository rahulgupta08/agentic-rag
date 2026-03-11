import os

from openai import embeddings

from app.ingestion.document_loader import DocumentLoader
from app.ingestion.chunker_new import DocumentChunker
from app.ingestion.ingestion_utils import generate_document_id , embedding_records_to_vectors
from app.embeddings.local_embedder import LocalEmbedder
from app.schemas.embedding_record import EmbeddingRecordSchema
from app.vectorstores.factory import get_vector_store
from app.ingestion.metadata_extractor import extract_metadata_from_filename


def load_and_chunk_document():


    collection = "financial_documents"

    loader = DocumentLoader(
        "data/raw_documents/aapl-10K_2023.pdf"
    )

    print("File path:", loader.file_path)
    document_metadata = extract_metadata_from_filename(loader.file_path)

    print("Extracted metadata:", document_metadata)
    

    document = loader.load()

    document_id = generate_document_id(
        "data/raw_documents/aapl-10K_2023.pdf"
    )

    chunker = DocumentChunker()

    chunks = chunker.chunk(
        document=document,
        document_id=document_id,
        collection=collection,
        document_metadata=document_metadata

    )

    print("Total chunks:", len(chunks))

    print(chunks[2].metadata)
    embedder = LocalEmbedder()


    text_for_embedding = [chunk.text for chunk in chunks]

    embeddings = embedder.embed_batch(text_for_embedding)
    print("Embeddings generated ", len(embeddings))

    embedding_records = []

    for chunk, vector in zip(chunks, embeddings):
        record = EmbeddingRecordSchema(
            id=chunk.id,
            vector=vector,
            text=chunk.text,
            metadata=chunk.metadata
        )
        embedding_records.append(record)

    

    print("Embeddings created:", len(embedding_records))
   
    vectors_for_db = embedding_records_to_vectors(embedding_records)

    print("vectors to be inserted : ", len(vectors_for_db))

    vector_store = get_vector_store()
    vector_store.upsert(vectors=vectors_for_db, namespace=collection)

    print("Inserted vectors:", len(vectors_for_db))

    

if __name__ == "__main__":
    load_and_chunk_document()