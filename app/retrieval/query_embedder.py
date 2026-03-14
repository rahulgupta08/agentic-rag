
from app.bootstrap import bootstrap
bootstrap()
import logging
logger = logging.getLogger(__name__)
from app.core.exceptions import EmbeddingError
from app.config import OPENAI_API_KEY, EMBEDDING_MODEL

try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    raise EmbeddingError(f"Failed to initialize OpenAI client: {str(e)}")


def embed_query(query: str) -> list[float]:
    if not query:
        raise EmbeddingError("Query cannot be empty")

    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query
        )

        embedding = response.data[0].embedding
        logger.info("Query embedding generated successfully")

        return embedding

    except Exception as e:
        logger.exception("Query embedding failed")
        raise EmbeddingError(f"Query embedding failed: {str(e)}")