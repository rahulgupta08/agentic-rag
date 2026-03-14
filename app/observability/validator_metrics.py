from app.bootstrap import bootstrap
bootstrap()
import logging
logger = logging.getLogger(__name__)          
class ValidatorMetrics:

    def log(self, score):

        logger.info(f"[AGENT_METRIC] validation_score={score}")