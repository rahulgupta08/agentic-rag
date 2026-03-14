from pathlib import Path
from typing import List
import json
from pydantic import BaseModel, Field
from app.logging.logger import logger



DATASET_PATH = Path("data/evaluation_dataset_v1.json")


class EvaluationSample(BaseModel):
    id: str
    question: str
    ground_truth: str
    expected_tools: List[str] = Field(default_factory=list)
    category: str
    difficulty: str


class DatasetBuilder:

    def __init__(self, dataset_path: Path = DATASET_PATH):
        self.dataset_path = dataset_path

    def load(self) -> List[EvaluationSample]:
        """
        Load evaluation dataset from JSON file.
        """

        logger.info(f"Loading evaluation dataset from JSON file {self.dataset_path} ")

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        with open(self.dataset_path, "r") as f:
            data = json.load(f)

        samples = [EvaluationSample(**item) for item in data]

        return samples

    def summary(self, samples: List[EvaluationSample]) -> None:
        """
        Print dataset statistics for debugging and monitoring.
        """
        logger.info("Dataset Summary")

        category_counts = {}
        tool_counts = {}
        difficulty_counts = {}

        for sample in samples:

            category_counts[sample.category] = (
                category_counts.get(sample.category, 0) + 1
            )

            difficulty_counts[sample.difficulty] = (
                difficulty_counts.get(sample.difficulty, 0) + 1
            )

            for tool in sample.expected_tools:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

       
        
        logger.info("-------------------")
        logger.info(f"Total samples: {len(samples)}")

        logger.info("Categories:")

        for k, v in category_counts.items():
            logger.info(f"{k}: {v}")

        logger.info("Tools:")
        for k, v in tool_counts.items():
            logger.info(f"{k}: {v}")

        logger.info("\nDifficulty:")
        for k, v in difficulty_counts.items():
            logger.info(f"{k}: {v}")