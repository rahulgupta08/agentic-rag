
from app.embeddings.local_embedder import LocalEmbedder
from app.embeddings.openai_embedder import OpenAIEmbedder
from app.vectorstores.factory import get_vector_store
from app.embeddings.embedder_factory import get_embedder





class RAGServiceBuilder:

    def __init__(
        self,
        vector_db: str = "pinecone",
        embedding: str = "local",
        top_k: int = 5
    ):
        self.vector_db = vector_db
        self.embedding = embedding
        self.top_k = top_k

    def _get_vector_store(self):

        return get_vector_store(
        provider=self.vector_db,
        # pass config if needed
    )

    def _get_embedder(self):

        return get_embedder(provider=self.embedding)

    def build(self):

        vector_store = self._get_vector_store()
        embedder = self._get_embedder()

        retriever = None
        # DenseRetriever(
        #     vector_store=vector_store,
        #     embedder=embedder,
        #     reranker=None,
        #     top_k=self.top_k
        # )

        llm =  None #ChatOpenAI(...)
        prompt_builder = None # RAGPromptBuilder()

        return None
    # RAGService(
    #         retriever=retriever,
    #         llm=llm,
    #         prompt_builder=prompt_builder
    #     )