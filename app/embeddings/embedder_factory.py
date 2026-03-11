from app.config import INGESTION_CONFIG

from .openai_embedder import OpenAIEmbedder
from .local_embedder import LocalEmbedder


def get_embedder():

    provider = INGESTION_CONFIG["embedding_provider"]

    if provider == "openai":

        return OpenAIEmbedder(
            model=INGESTION_CONFIG["embedding_model"]
        )

    elif provider == "local":

        return LocalEmbedder(
            model_name=INGESTION_CONFIG["local_embedding_model"]
        )

    else:
        raise ValueError("Unsupported embedding provider")