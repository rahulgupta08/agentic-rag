import asyncio
import json
from pathlib import Path

from app.evaluation.offline_runner import OfflineEvaluationRunner
from app.evaluation.context_extractor import ContextExtractor
from app.evaluation.ragas_evaluator import RagasEvaluator
from app.evaluation.evaluation_dashboard import EvaluationDashboard
from app.logging.logger import logger



RESULTS_PATH = Path("data/evaluation_results.json")
RAGAS_REPORT_PATH = Path("data/ragas_report.json")


async def main():

    runner = OfflineEvaluationRunner()

    logger.info("===== Running Agent Evaluation =====\n")

    results = await runner.run()

   
    logger.info(f"Total samples processed: {len(results)}")

    # Save raw results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2,default=str)

    logger.info(f"Raw evaluation results saved to: {RESULTS_PATH}")

    # Extract RAGAS dataset
    extractor = ContextExtractor()

    ragas_samples = extractor.build_ragas_dataset(results)

    logger.info(f"RAGAS samples prepared: {len(ragas_samples)}")

    # Run RAGAS evaluation
    evaluator = RagasEvaluator()

    ragas_metrics = evaluator.evaluate(ragas_samples)

    # Save RAGAS report
    with open(RAGAS_REPORT_PATH, "w") as f:
        json.dump(ragas_metrics, f, indent=2, default=str)

    logger.info(f"RAGAS report saved to: {RAGAS_REPORT_PATH}")

    # Dashboard metrics
    dashboard = EvaluationDashboard(results)

    tool_accuracy = dashboard.tool_accuracy()

    retry_rate, avg_retry = dashboard.retry_metrics()

    difficulty_perf = dashboard.difficulty_performance()

    logger.info("===== RAGAS Metrics =====")
    logger.info(f"Ragas Metrics: {ragas_metrics}")

    logger.info("===== Agent Metrics =====")

    logger.info(f"Tool Selection Accuracy: {tool_accuracy:.2f}")

    logger.info(f"Retry Rate: {retry_rate:.2f}")

    logger.info(f"Average Retry Count: {avg_retry:.2f}")

    logger.info("===== Difficulty Performance =====")

    for difficulty, score in difficulty_perf.items():
        logger.info(f"{difficulty}: {score:.2f}")

    await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())