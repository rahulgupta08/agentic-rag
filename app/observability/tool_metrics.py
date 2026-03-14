from app.bootstrap import bootstrap
bootstrap()
import logging
logger = logging.getLogger(__name__) 

class ToolMetrics:

    def log(self, tool_name):

        logger.info(f"[AGENT_METRIC] tool_used={tool_name}")