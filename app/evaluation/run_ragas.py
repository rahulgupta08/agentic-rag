from app.evaluation.dataset_loader import DatasetLoader
from app.evaluation.experiment_runner import ExperimentRunner
from app.evaluation.metrics_store import MetricsStore

loader = DatasetLoader("test_dataset.json")
dataset = loader.load()

runner = ExperimentRunner(rag_service)
results = runner.run(dataset)

store = MetricsStore()
store.save("baseline_experiment", results)

print(results)