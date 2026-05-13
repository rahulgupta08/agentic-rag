from app.embeddings.local_embedder import LocalEmbedder
from app.embeddings.openai_embedder import OpenAIEmbedder
from app.vectorstores.factory import get_vector_store
from app.embeddings.embedder_factory import get_embedder
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.retriever_pipeline import RetrieverPipeline
from app.prompts.rag_prompt import RAGPromptBuilder
from app.llm.llm_generator_factory import LLMGeneratorFactory
from app.query.rewriting.llm_query_rewriter import LLMQueryRewriter
from app.query.expansion.llm_query_expander import LLMQueryExpander
from app.rerankers.cross_encoder_reranker import CrossEncoderReranker
from app.guardrails.guardrail_manager import GuardrailManager
from langchain_openai import ChatOpenAI
from app.config import OPENAI_API_KEY


class RAGServiceBuilder:

    def __init__(
        self,
        vector_db: str = "pinecone",
        embedding: str = "local",
        top_k: int = 5,
        enable_reranking: bool = True,
        enable_query_rewriting: bool = False,
        enable_guardrails: bool = False
    ):
        self.vector_db = vector_db
        self.embedding = embedding
        self.top_k = top_k
        self.enable_reranking = enable_reranking
        self.enable_query_rewriting = enable_query_rewriting
        self.enable_guardrails = enable_guardrails

    def _get_vector_store(self):
        return get_vector_store(provider=self.vector_db)

    def _get_embedder(self):
        return get_embedder(provider=self.embedding)
    
    def _get_reranker(self):
        if not self.enable_reranking:
            return None
        
        return CrossEncoderReranker()
    
    def _get_query_transformer(self):
        if not self.enable_query_rewriting:
            return None
        
        llm = LLMGeneratorFactory.create(provider="openai")
        return LLMQueryRewriter(llm=llm)
    
    def _get_query_expander(self):
        if not self.enable_query_rewriting:
            return None
        
        llm = LLMGeneratorFactory.create(provider="openai")

        return LLMQueryExpander(llm)
    
    def _get_guardrail_manager(self):
        if not self.enable_guardrails:
            return None
        
        return GuardrailManager(
            domain_keywords=["revenue", "financial", "stock", "earnings"],
            blocked_terms=["hack", "exploit", "bypass"]
        )

    def build(self):
        """Build and return configured RAGService."""
        
        # Core components
        vector_store = self._get_vector_store()
        embedder = self._get_embedder()
        
        # Optional components
        reranker = self._get_reranker()
        query_transformer = self._get_query_transformer()
        query_expander = self._get_query_expander()
        guardrail_manager = self._get_guardrail_manager()
        
        # Build retriever pipeline
        base_retriever = DenseRetriever(
            vector_store=vector_store,
            embedder=embedder,
            top_k=self.top_k
        )
        
        retriever = RetrieverPipeline(
            base_retriever=base_retriever,
            reranker=reranker,
            query_transformer=query_transformer,
            query_expander=query_expander,
            confidence_threshold=0.5
        )
        
        # Build LLM and prompt builder
        llm = LLMGeneratorFactory.create(provider="deepseek")
        prompt_builder = RAGPromptBuilder()
        
        # Build RAGService
        from app.services.rag_service import RAGService
        return RAGService(
            retriever=retriever,
            llm=llm,
            prompt_builder=prompt_builder,
            query_expander=query_expander,
            guardrail_manager=guardrail_manager
        )