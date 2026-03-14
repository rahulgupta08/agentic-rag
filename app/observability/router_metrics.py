from app.bootstrap import bootstrap
bootstrap()
import logging
logger = logging.getLogger(__name__)
class RouterMetrics:

    def log(self, decision):

        logger.info(f"[AGENT_METRIC] router_decision={decision}")