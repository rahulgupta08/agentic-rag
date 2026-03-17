from dataclasses import dataclass


@dataclass
class RAGInput:
    query: str


@dataclass
class AgentInput:
    query: str
    session_id: str


@dataclass
class IngestionInput:
    file_path: str
    collection: str = "financial_documents"


@dataclass
class EvaluationInput:
    # placeholder for future extensibility
    pass