import os
import uuid
from typing import List
from app.vectorstores.base_store import BaseVectorStore


class DocumentIngestor:

    def __init__(self, vector_store: BaseVectorStore, embedder, chunker):
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunker = chunker


    def ingest_text(self, text: str, source: str = None):

        # 1️⃣ Chunk
        chunks = self.chunker.chunk(text)

        if not chunks:
            return

        # 2️⃣ Embed

        print("Embedder " , type(self.embedder))

        embeddings = self.embedder.embed_batch(chunks)

        # 3️⃣ Prepare standard vector format
        vectors = []

        for chunk, embedding in zip(chunks, embeddings):

            vectors.append({
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "text": chunk,
                "metadata": {
                    "source": source
                }
            })

        # 4️⃣ Upsert (store-agnostic)
        self.vector_store.upsert(vectors)


    def ingest_file(self, file_path: str, loader):

        print("Ingest file", file_path)

        text = loader(file_path)

        source_name = os.path.basename(file_path)
        print("souce name" , source_name)

        self.ingest_text(text, source=source_name)