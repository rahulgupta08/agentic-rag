from app.query.base_query_transformer import BaseQueryTransformer
from app.query.query_validator import QueryValidator
import logging
import re
logger = logging.getLogger(__name__)


class LLMQueryExpander(BaseQueryTransformer):

    def __init__(self, llm, num_queries=3):
        self.llm = llm
        self.num_queries = num_queries
        self.validator = QueryValidator()

    def transform(self, query: str):

        logging.info(f"LLM Query Expander query {query}")
        prompt = self._build_prompt(query)
        logging.info(f"LLM Query Expander prompt {prompt}")
        
        response = self.llm.generate(query=query, context=prompt)
        return self._parse_response(response, query)

    async def transform_async(self, query: str):
        """Asynchronous transform for Agent mode"""
        prompt = self._build_prompt(query)
        response = await self.llm.agenerate(query=query, context=prompt)
        return self._parse_response(response, query)

    def _build_prompt(self, query: str) -> str:
        """Helper method to build the expansion prompt"""
        return f"""
        Create {self.num_queries} semantic variations of the search query that:
        1. Maintain the same information-seeking goal
        2. Use alternative but equivalent business/financial terms
        3. Keep all company names and document types unchanged
        4. Focus on finding the same information in different ways

        Examples:
        Original: "What are Apple's revenue numbers in their 10-K?"
        Variations:
        - Find Apple's reported revenue figures in their 10-K filing
        - Show Apple's income data from their 10-K report
        - Extract Apple's earnings information from 10-K document

        Original: "List risk factors in Tesla's annual report"
        Variations:
        - Identify key risks disclosed in Tesla's annual report
        - Find risk disclosures from Tesla's annual filing
        - Extract risk assessment sections from Tesla's annual report

        Original Query: {query}
        
        Semantic Variations (maintain same intent, use equivalent terms):
        """

    def _parse_response(self, response: str, original_query: str) -> list:
        """
        Helper method to parse LLM response into queries while validating
        semantic variations.
        """
        # Split response into lines and clean up
        lines = [
            line.strip("- ").strip()
            for line in response.split("\n")
            if line.strip() and not line.startswith("Original")
        ]
        
        # Filter valid variations
        valid_variations = []
        for variation in lines:
            # Skip empty or too short variations
            if not variation or len(variation) < 10:
                continue
                
            # Validate against original query
            if self.validator.validate_rewrite(original_query, variation):
                valid_variations.append(variation)
                logger.info("Valid variation: %s", variation)
            else:
                logger.warning("Invalid variation rejected: %s", variation)
                
        # If no valid variations, return original query
        if not valid_variations:
            logger.warning("No valid variations found, using original query")
            return [original_query]
            
        return valid_variations[:self.num_queries]