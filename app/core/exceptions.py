from typing import Optional


class RAGException(Exception):
    """
    Base exception for the RAG application.
    """

    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code or "RAG_ERROR"
        super().__init__(self.message)


# -------------------------
# Document / Ingestion Errors
# -------------------------

class DocumentProcessingError(RAGException):
    def __init__(self, message: str):
        super().__init__(message, error_code="DOCUMENT_PROCESSING_ERROR")


class ChunkingError(RAGException):
    def __init__(self, message: str):
        super().__init__(message, error_code="CHUNKING_ERROR")


class EmbeddingError(RAGException):
    def __init__(self, message: str):
        super().__init__(message, error_code="EMBEDDING_ERROR")


class VectorStoreError(RAGException):
    def __init__(self, message: str):
        super().__init__(message, error_code="VECTOR_STORE_ERROR")


# -------------------------
# Retrieval Errors
# -------------------------

class RetrievalError(RAGException):
    def __init__(self, message: str):
        super().__init__(message, error_code="RETRIEVAL_ERROR")


# -------------------------
# LLM Errors
# -------------------------

class LLMGenerationError(RAGException):
    def __init__(self, message: str):
        super().__init__(message, error_code="LLM_GENERATION_ERROR")