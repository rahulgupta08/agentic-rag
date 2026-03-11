
import logging
from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.core.exceptions import ChunkingError
from typing import List



logger = logging.getLogger(__name__)

class ChunkingClass:
    """
    Supports multiple chunking strategies:
    - recursive
    - token
    """

    def __init__(
        self,
        strategy: str = "text",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # -------------------------
    # Public API
    # -------------------------

    def chunk(self, text: str) -> List[str]:
        logger.info("Inside Chunk")
        if not text:
            raise ChunkingError("Cannot chunk empty text")

        try:
            if self.strategy == "recursive":
                return self._recursive_chunk(text)

            elif self.strategy == "token":
                return self._token_chunk(text)
            
            elif self.strategy == "text":
                return self._text_chunk(text)

            else:
                raise ChunkingError(f"Unknown chunking strategy: {self.strategy}")

        except Exception as e:
            logger.exception("Chunking failed")
            raise ChunkingError(str(e))

    # -------------------------
    # Recursive Character Chunking
    # -------------------------

    def _recursive_chunk(self, text: str) -> List[str]:
        logger.info("Using recursive chunking")

        separators = ["\n\n", "\n", ". ", " ", ""]

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to split cleanly using separators
            for sep in separators:
                if sep in chunk:
                    split_pos = chunk.rfind(sep)
                    if split_pos != -1:
                        end = start + split_pos + len(sep)
                        chunk = text[start:end]
                        break

            chunks.append(chunk.strip())
            start = end - self.chunk_overlap

        logger.info(f"Created {len(chunks)} chunks using recursive strategy")
        return chunks

    # -------------------------
    # Token-Based Chunking
    # -------------------------

    def _token_chunk(self, text: str) -> List[str]:
        logger.info("Using token-based chunking")

        try:
            import tiktoken  # tokenizer used by OpenAI models
        except ImportError:
            raise ChunkingError(
                "tiktoken not installed. Install via: pip install tiktoken"
            )

        encoding = tiktoken.get_encoding("cl100k_base")

        tokens = encoding.encode(text)

        chunks = []
        start = 0
        total_tokens = len(tokens)

        while start < total_tokens:
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]

            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text.strip())

            start = end - self.chunk_overlap

        logger.info(f"Created {len(chunks)} chunks using token strategy")
        return chunks

    def _text_chunk(self,text: str) -> list[str]:
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP

        logger.info(f"Generated {len(chunks)} chunks")
        return chunks
