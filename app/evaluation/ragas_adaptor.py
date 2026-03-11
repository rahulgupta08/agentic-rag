from typing import Dict, List


class RagasAdapter:

    @staticmethod
    def transform(
        question: str,
        rag_output: Dict,
        ground_truth: str = None
    ) -> Dict:

        contexts = [
            doc["text"] for doc in rag_output.get("documents_used", [])
        ]

        return {
            "question": question,
            "answer": rag_output.get("answer"),
            "contexts": contexts,
            "ground_truth": ground_truth
        }