from app.bootstrap import bootstrap
bootstrap()

import logging

logger = logging.getLogger(__name__)


class EvaluationDashboard:

    def __init__(self, results):
        self.results = results

    def tool_accuracy(self):
        """
        Compare expected tools vs actual tools used by the agent.
        """

        logger.info("Evaluating Tool Accuracy")
        correct = 0
        total = 0

        for r in self.results:

            if "error" in r:
                continue

            expected = set(r.get("expected_tools", []))

            actual = set(
                t.get("tool")
                for t in r.get("tool_results", [])
                if isinstance(t, dict) and "tool" in t
            )

            if expected == actual:
                correct += 1

            total += 1

        return correct / total if total else 0

    def retry_metrics(self):
        """
        Compute retry statistics from validator behavior.
        """

        logger.info("Evaluating Retry Metrics")

        retries = [
            r.get("retry_count", 0)
            for r in self.results
            if "error" not in r
        ]

        if not retries:
            return 0, 0

        retry_rate = sum(1 for r in retries if r > 0) / len(retries)
        avg_retry = sum(retries) / len(retries)

        return retry_rate, avg_retry

    def difficulty_performance(self):
        """
        Measure performance grouped by difficulty level.
        """

        logger.info("Evaluating Difficulty Performance")

        stats = {}

        for r in self.results:

            if "error" in r:
                continue

            difficulty = r.get("difficulty", "unknown")

            stats.setdefault(difficulty, {"total": 0, "correct": 0})

            stats[difficulty]["total"] += 1

            if r.get("answer"):
                stats[difficulty]["correct"] += 1

        performance = {}

        for diff, s in stats.items():
            performance[diff] = s["correct"] / s["total"]

        return performance