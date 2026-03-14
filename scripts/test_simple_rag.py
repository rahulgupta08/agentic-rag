from app.bootstrap import bootstrap
bootstrap()
import logging
logger = logging.getLogger(__name__)
from app.rag.simple_rag import SimpleRAG

def main():

    rag = SimpleRAG()

    query = "What risk factors does Apple mention in the 10-K?"

    logger.info(f"\nQUERY:\n{query}")


    response = rag.ask(question=query, top_k=10)

    logger.info(f"\nANSWER:\n{response['answer']}")


    logger.info("\nRETRIEVED DOCUMENTS:")

    

if __name__ == "__main__":
    main()