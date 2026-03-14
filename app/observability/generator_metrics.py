from app.bootstrap import bootstrap
bootstrap()
import logging
logger = logging.getLogger(__name__)

class GeneratorMetrics:

    def log(self, size):

        logger.info(f"[AGENT_METRIC] context_chunks={size}")