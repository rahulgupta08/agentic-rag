from openai import OpenAI

from app.schemas.embedding_record import EmbeddingRecordSchema
from app.config import INGESTION_CONFIG
from dotenv import load_dotenv

load_dotenv()


class Embedder:

    def __init__(self):

        self.client = OpenAI()

        self.model = INGESTION_CONFIG["embedding_model"]

    def embed(self, chunks):

        texts = [chunk.text for chunk in chunks]

        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )

        embeddings = []

        for chunk, emb in zip(chunks, response.data):

            record = EmbeddingRecordSchema(
                id=chunk.id,
                vector=emb.embedding,
                text=chunk.text,
                metadata=chunk.metadata
            )

            embeddings.append(record)

        return embeddings