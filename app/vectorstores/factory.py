from app.config import VECTOR_DB_PROVIDER, PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_DIMENSION
from app.vectorstores.weaviate_store import WeaviateStore
from app.vectorstores.pinecone_store import PineconeStore
from app.core.exceptions import EmbeddingError
from app.config import WEAVIATE_COLLECTION
import logging
import weaviate
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from app.embeddings.embedder_factory import get_embedder
import sys

load_dotenv()

logger = logging.getLogger(__name__)


def get_vector_store():

    embedder = get_embedder()
    dimension = embedder.dimension

    print('DIMENTION : -------- ', dimension, file= sys.stderr)
    print('VECTOR_DB_PROVIDER' , VECTOR_DB_PROVIDER, file = sys.stderr)

    if VECTOR_DB_PROVIDER == "pinecone":
        pc = Pinecone(api_key=PINECONE_API_KEY)

        existing_indexes = [idx["name"] for idx in pc.list_indexes()]

        if PINECONE_INDEX_NAME not in existing_indexes:

            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        index = pc.Index(PINECONE_INDEX_NAME)

        return PineconeStore(index=index)

    elif VECTOR_DB_PROVIDER == "weaviate":
        client = weaviate.connect_to_local(
                    host="localhost",
                    port=8080
                )       
        return WeaviateStore(
            collection_name= WEAVIATE_COLLECTION,
            client=client
        )

    else:
        raise ValueError("Unsupported vector database provider")