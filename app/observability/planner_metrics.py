from app.bootstrap import bootstrap
bootstrap()
import logging
logger = logging.getLogger(__name__)

class PlannerMetrics:

    def log(self, plan):

        tools = []

        tools = [step.get("tool") for step in plan]


        logger.info(f"[AGENT_METRIC] plan_steps={len(plan)}")
        logger.info(f"[AGENT_METRIC] planned_tools={tools}")