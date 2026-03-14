from app.logging.logger import logger

class ToolMetrics:

    def log(self, tool_name):

        logger.info(f"[AGENT_METRIC] tool_used={tool_name}")