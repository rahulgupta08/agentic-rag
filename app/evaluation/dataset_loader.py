import json
from typing import List, Dict


class DatasetLoader:

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Dict]:
        """
        Expected JSON format:

        [
            {
                "question": "...",
                "ground_truth": "..."
            }
        ]
        """

        with open(self.file_path, "r") as f:
            data = json.load(f)

        return data