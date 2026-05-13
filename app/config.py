import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"#text-embedding-3-large"
EMBEDDING_DIMENSION = 768
LLM_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
VECTOR_DB_PROVIDER : str = os.getenv("VECTOR_DB_PROVIDER", 'weaviate')#'pinecone' #pinecone
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
WEAVIATE_URL: str = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_COLLECTION: str = os.getenv("WEAVIATE_COLLECTION", "financial_documents") #sampleessay2

TOP_K = 5

INGESTION_CONFIG = {
    "embedding_provider":"local" ,  #or "openai"
    "embedding_model": "text-embedding-3-large",
    "local_embedding_model": "BAAI/bge-base-en-v1.5",
    "chunk_size": 800,
    "chunk_overlap": 120
}

# Validate required keys
if not PINECONE_API_KEY and VECTOR_DB_PROVIDER == "pinecone":
    raise ValueError("PINECONE_API_KEY environment variable not set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")
if not OPENROUTER_API_KEY:
       raise ValueError("OPENAI_API_KEY environment variable not set")

# text-embedding-3-large	3072
# text-embedding-3-small	1536
# BAAI/bge-base-en-v1.5	768
# intfloat/e5-large-v2	1024