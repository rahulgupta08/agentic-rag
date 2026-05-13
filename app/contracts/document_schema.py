from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Document:
    """Represents a document chunk with metadata."""
    content: str
    metadata: Dict[str, Any]
    id: str

@dataclass
class SearchResult:
    """Represents a search result from the retriever."""
    documents: List[Document]
    query: str