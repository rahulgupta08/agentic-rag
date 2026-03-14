from app.bootstrap import bootstrap
bootstrap()
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)



class ContextExtractor:
    """
    Extracts contexts from agent outputs in a tool-agnostic way.

    Supports:
    - vector_search results
    - web_search results
    - future tools
    """

    def extract_contexts(self, result: Dict[str, Any]) -> List[str]:
        """
        Extract textual contexts from agent execution result.
        """

        logger.info("Extracting Context ")

        contexts: List[str] = []

        # 1. Prefer retrieved_docs if available
        retrieved_docs = result.get("retrieved_docs", [])

        for doc in retrieved_docs:

            if isinstance(doc, str):
                contexts.append(doc)

            elif isinstance(doc, dict):
                text = doc.get("page_content") or doc.get("content")

                if text:
                    contexts.append(text)

            elif hasattr(doc, "page_content"):
                contexts.append(doc.page_content)

        # 2. Extract contexts from tool_results
        tool_results = result.get("tool_results", [])

        for tool_result in tool_results:

            if not isinstance(tool_result, dict):
                continue

            output = tool_result.get("output")

            if isinstance(output, str):
                contexts.append(output)

            elif isinstance(output, list):

                for item in output:

                    if isinstance(item, str):
                        contexts.append(item)

                    elif isinstance(item, dict):

                        text = (
                            item.get("content")
                            or item.get("text")
                            or item.get("snippet")
                        )

                        if text:
                            contexts.append(text)

        # Remove duplicates while preserving order
        seen = set()
        unique_contexts = []

        for ctx in contexts:

            if ctx not in seen:
                unique_contexts.append(ctx)
                seen.add(ctx)

        return unique_contexts

    def build_ragas_sample(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert agent output into RAGAS dataset format.
        """

        logger.info("Building Ragas Sample")
        contexts = self.extract_contexts(result)

        return {
            "question": result.get("question"),
            "answer": result.get("answer"),
            "contexts": contexts,
            "ground_truth": result.get("ground_truth"),
        }

    def build_ragas_dataset(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert evaluation results into a dataset for RAGAS evaluation.
        """

        logger.info("Building Ragas Dataset")

        dataset = []

        for r in results:

            if "error" in r:
                continue

            sample = self.build_ragas_sample(r)

            dataset.append(sample)

        return dataset