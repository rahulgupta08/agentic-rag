from app.ingestion.chunker import ChunkingClass
from app.config import CHUNK_SIZE, CHUNK_OVERLAP,EMBEDDING_MODEL
from scripts.ingest_document import DocumentIngestor
from app.vectorstores.factory import get_vector_store
from app.embeddings.openai_embedder import OpenAIEmbedder
from app.config import LLM_MODEL
from app.ingestion.loader import load_pdf
import sys
import logging


logger = logging.getLogger(__name__)

def main(file_path: str):

    chunker = ChunkingClass(
        strategy="text",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    embedder = OpenAIEmbedder(model=EMBEDDING_MODEL)

    vector_store = get_vector_store()  # returns WeaviateStore or PineconeStore

    ingestor = DocumentIngestor(
        vector_store=vector_store,
        embedder=embedder,
        chunker=chunker
    )

    ingestor.ingest_file(file_path=file_path,loader=load_pdf)

if __name__ == "__main__":
    main(sys.argv[1])

