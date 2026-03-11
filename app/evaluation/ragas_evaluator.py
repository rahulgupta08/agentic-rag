from typing import List
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset


class RagasEvaluator:

    def evaluate(self, samples: List[dict]) -> dict:
        """
        samples must be list of dicts:
        {
            "question": "...",
            "answer": "...",
            "contexts": [...],
            "ground_truth": "..."
        }
        """

        dataset = Dataset.from_list(samples)

        results = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        )

        return results