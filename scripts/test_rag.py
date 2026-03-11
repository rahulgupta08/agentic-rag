from app.retrieval.dense_retriever import DenseRetriever
from app.embeddings.openai_embedder import OpenAIEmbedder
from app.vectorstores.factory import get_vector_store
from app.services.rag_service import RAGService
from app.retrieval.llm_reranker import LLMReranker
from app.config import EMBEDDING_MODEL
from openai import OpenAI
from app.retrieval.cross_encoder_reranker import CrossEncoderReranker
from app.queryexpansion.llm_query_expander import LLMQueryExpander



def main():

    embedder = OpenAIEmbedder(model=EMBEDDING_MODEL)
    vector_store = get_vector_store()
    


    llm = OpenAI()

    reranker = LLMReranker(llm)

    #query_expander = LLMQueryExpander(llm)

    #reranker = CrossEncoderReranker(
     #           model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
      #          )

    retriever = DenseRetriever(
        vector_store=vector_store,
        embedder=embedder,
        reranker=reranker
    )

    rag_service = RAGService(
        retriever=retriever,
        llm=llm,
        #query_expander=query_expander
    )

    response = rag_service.generate("", top_k=5)
    print(response)

if __name__ == "__main__":
    main()

