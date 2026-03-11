from app.evaluation.ragas_adaptor import RagasAdapter
from app.evaluation.ragas_evaluator import RagasEvaluator

from typing import List



class ExperimentRunner:

    def __init__(self, rag_service):
        self.rag_service = rag_service
        self.evaluator = RagasEvaluator()

    def run(self, dataset: List[dict]) -> dict:

        samples = []

        for item in dataset:

            question = item["question"]
            ground_truth = item.get("ground_truth")

            rag_output = self.rag_service.generate(question)

            sample = RagasAdapter.transform(
                question=question,
                rag_output=rag_output,
                ground_truth=ground_truth
            )

            samples.append(sample)

        results = self.evaluator.evaluate(samples)

        return results