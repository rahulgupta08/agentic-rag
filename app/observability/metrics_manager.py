from .retrieval_metrics import RetrievalMetrics
from .retry_metrics import RetryMetrics
from .validator_metrics import ValidatorMetrics
from .tool_metrics import ToolMetrics
from .generator_metrics import GeneratorMetrics
from .planner_metrics import PlannerMetrics
from .router_metrics import RouterMetrics


class MetricsManager:

    def __init__(self):

        self.retrieval = RetrievalMetrics()
        self.retry = RetryMetrics()
        self.validator = ValidatorMetrics()
        self.tool = ToolMetrics()
        self.planner = PlannerMetrics()
        self.router = RouterMetrics()
        self.generator = GeneratorMetrics()


    def log_retrieval(self, docs):
        self.retrieval.log(docs)

    def log_retry(self, retry_count):
        self.retry.log(retry_count)

    def log_validation(self, score):
        self.validator.log(score)

    def log_tool(self, tool_name):
        self.tool.log(tool_name)

    def log_plan(self, plan):
        self.planner.log(plan)

    def log_router_decision(self, decision):
        self.router.log(decision)

    def log_context_size(self, size):
        self.generator.log(size)