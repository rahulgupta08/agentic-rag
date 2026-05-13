from typing import Protocol, List, Dict, Any, Optional
from app.contracts.document_schema import Document, SearchResult

class RetrieverProtocol(Protocol):
    """Protocol for retriever services that agent nodes depend on."""
    
    async def retrieve(self, query: str, top_k: int = 5) -> SearchResult:
        """Retrieve relevant documents for a query."""
        ...

class LLMProtocol(Protocol):
    """Protocol for LLM services that agent nodes depend on."""
    
    async def ainvoke(self, prompt: str, **kwargs) -> Any:
        """Invoke the LLM with a prompt."""
        ...

class PromptBuilderProtocol(Protocol):
    """Protocol for prompt builders that agent nodes depend on."""
    
    def build_prompt(self, query: str, context: str) -> str:
        """Build a prompt for the LLM."""
        ...

class QueryExpanderProtocol(Protocol):
    """Protocol for query expanders that agent nodes depend on."""
    
    async def expand(self, query: str) -> str:
        """Expand a query to improve retrieval."""
        ...

class GuardrailManagerProtocol(Protocol):
    """Protocol for guardrail managers that agent nodes depend on."""
    
    def check_input_guardrail(self, query: str) -> Dict[str, Any]:
        """Check if a query passes input guardrails."""
        ...
    
    def check_output_guardrail(self, answer: str) -> Dict[str, Any]:
        """Check if an answer passes output guardrails."""
        ...

class RAGServiceProtocol(Protocol):
    """Protocol for RAG service that agent nodes depend on."""
    
    # Methods from RAGService that agent nodes use
    async def generate(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Generate an answer for a query."""
        ...
    
    async def generate_from_context(self, query: str, context: str) -> Dict[str, Any]:
        """Generate an answer from provided context."""
        ...
    
    async def generate_iterative_rag(self, query: str, top_k: int = 5, max_iterations: int = 2) -> Dict[str, Any]:
        """Generate an answer using iterative RAG."""
        ...
    
    # Properties from RAGService that agent nodes use
    retriever: RetrieverProtocol
    llm: LLMProtocol
    prompt_builder: PromptBuilderProtocol
    query_expander: QueryExpanderProtocol
    guardrail_manager: GuardrailManagerProtocol