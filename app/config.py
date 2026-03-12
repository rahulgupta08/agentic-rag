import os


PINECONE_API_KEY = ''
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"#text-embedding-3-large"
EMBEDDING_DIMENSION = 768
LLM_MODEL = "gpt-4o-mini"
OPENAI_API_KEY=''
VECTOR_DB_PROVIDER : str = 'weaviate'#'pinecone' #pinecone
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
WEAVIATE_URL: str = "http://localhost:8080"
WEAVIATE_COLLECTION: str = "financial_documents" #sampleessay2

TOP_K = 5

INGESTION_CONFIG = {
    "embedding_provider":"local" ,  #or "openai"
    "embedding_model": "text-embedding-3-large",
    "local_embedding_model": "BAAI/bge-base-en-v1.5",
    "chunk_size": 800,
    "chunk_overlap": 120
}


# text-embedding-3-large	3072
# text-embedding-3-small	1536
# BAAI/bge-base-en-v1.5	768
# intfloat/e5-large-v2	1024