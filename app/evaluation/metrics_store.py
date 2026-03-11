import json
from datetime import datetime


class MetricsStore:

    def __init__(self, file_path: str = "ragas_metrics.json"):
        self.file_path = file_path

    def save(self, experiment_name: str, results: dict):

        record = {
            "experiment_name": experiment_name,
            "timestamp": datetime.utcnow().isoformat(),
            "results": dict(results)
        }

        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
        except:
            data = []

        data.append(record)

        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=4)