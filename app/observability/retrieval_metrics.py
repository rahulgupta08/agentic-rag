from app.logging.logger import logger


class RetrievalMetrics:

    def log(self, docs):

        doc_count = len(docs)

        logger.info(f"[AGENT_METRIC] retrieved_docs={doc_count}")


        if doc_count == 0:
            logger.warning("[AGENT_METRIC] retrieval returned 0 documents")