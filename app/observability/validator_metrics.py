from app.logging.logger import logger

class ValidatorMetrics:

    def log(self, score):

        logger.info(f"[AGENT_METRIC] validation_score={score}")