from app.logging.logger import logger

class PlannerMetrics:

    def log(self, plan):

        tools = []

        tools = [step.get("tool") for step in plan]


        logger.info(f"[AGENT_METRIC] plan_steps={len(plan)}")
        logger.info(f"[AGENT_METRIC] planned_tools={tools}")