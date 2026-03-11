import tiktoken

from app.schemas.document_chunk import DocumentChunkSchema
from app.config import INGESTION_CONFIG

# What This Chunker Produces

# Example chunk:

# DocumentChunk(
#  id="apple_10k_2024_chunk_0042",
#  text="Apple's net sales increased significantly due to strong iPhone demand...",
#  metadata={
#    "document_id": "apple_10k_2024",
#    "collection": "financial_documents",
#    "page": 42,
#    "source": "data/raw_documents/apple_10k_2024.pdf"
#  }
# )


class DocumentChunker:

    def __init__(self):

        self.chunk_size = INGESTION_CONFIG["chunk_size"]
        self.chunk_overlap = INGESTION_CONFIG["chunk_overlap"]

        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk(self, document: dict, document_id: str, collection: str, document_metadata):

        chunks = []

        chunk_index = 0

        for page in document["pages"]:

            page_number = page["page_number"]
            text = page["text"]

            tokens = self.tokenizer.encode(text)

            start = 0

            while start < len(tokens):

                end = start + self.chunk_size

                chunk_tokens = tokens[start:end]

                chunk_text = self.tokenizer.decode(chunk_tokens)

                chunk_id = f"{document_id}_chunk_{chunk_index:04d}"

                # metadata = {
                #     "document_id": document_id,
                #     "collection": collection,
                #     "page": str(page_number),
                #     "source": document["file_path"],
                #     "chunk_index": str(chunk_index)

                # }

                metadata = {
                    "document_id": document_id,
                    "collection": collection,
                    "page": str(page_number),
                    "chunk_index": str(chunk_index),
                    "source": document["file_path"],
                    **document_metadata
                }

                chunk = DocumentChunkSchema(
                    id=chunk_id,
                    text=chunk_text,
                    metadata=metadata
                )

                chunks.append(chunk)

                chunk_index += 1

                start += self.chunk_size - self.chunk_overlap

        return chunks