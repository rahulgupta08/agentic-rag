from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict, Any
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from app.logging.logger import logger
from langchain_community.embeddings import HuggingFaceEmbeddings



class RagasEvaluator:

    def __init__(self):

        # Use same embedding model family as your retrieval
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5"
        )

    def _to_hf_dataset(self, ragas_samples: List[Dict[str, Any]]) -> Dataset:

        logger.info("Converting extracted samples to HuggingFace Dataset")

        dataset_dict = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        for sample in ragas_samples:
            dataset_dict["question"].append(sample["question"])
            dataset_dict["answer"].append(sample["answer"])
            dataset_dict["contexts"].append(sample["contexts"])
            dataset_dict["ground_truth"].append(sample["ground_truth"])

        return Dataset.from_dict(dataset_dict)

    def evaluate(self, ragas_samples: List[Dict[str, Any]]):

        logger.info("Running RAGAS evaluation")

        if not ragas_samples:
            raise ValueError("No samples provided")

        dataset = self._to_hf_dataset(ragas_samples)

        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            embeddings=self.embeddings,
        )

        logger.info(f"RAGAS RESULT: {result}")

        return result