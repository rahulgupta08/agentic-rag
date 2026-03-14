from app.logging.logger import logger

class RouterMetrics:

    def log(self, decision):

        logger.info(f"[AGENT_METRIC] router_decision={decision}")