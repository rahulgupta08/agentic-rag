from app.bootstrap import bootstrap
bootstrap()
import logging
logger = logging.getLogger(__name__)

class RetryMetrics:

    def log(self, retry_count):

         logger.warning(f"[AGENT_METRIC] retry_triggered retry_count={retry_count}")