
from app.bootstrap import bootstrap
bootstrap()
import logging
logger = logging.getLogger(__name__)

from app.rag.advance_rag import AdvancedRAG

rag = AdvancedRAG(similarity_threshold=0.5)

response = rag.ask("when is Independence day celebrated in India?", "hybrid")

logger.info(f"Confidence: {response['confidence']}\nAnswer: {response['answer']}")
