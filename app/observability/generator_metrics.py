from app.logging.logger import logger

class GeneratorMetrics:

    def log(self, size):

        logger.info(f"[AGENT_METRIC] context_chunks={size}")