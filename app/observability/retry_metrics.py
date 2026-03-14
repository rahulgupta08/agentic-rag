from app.logging.logger import logger

class RetryMetrics:

    def log(self, retry_count):

         logger.warning(f"[AGENT_METRIC] retry_triggered retry_count={retry_count}")