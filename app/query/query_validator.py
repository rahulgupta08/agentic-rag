import re
import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

class QueryValidator:
    """
    Validates rewritten queries to ensure they preserve the original intent
    and key elements of the query.
    """

    def __init__(self):
        self.meta_query_patterns = [
            r"generate.*quer(y|ies)",
            r"create.*search",
            r"list.*questions",
            r"suggest.*topics"
        ]
        
        self.key_entities = {
            "companies": {
                "apple", "google", "microsoft", "amazon", "facebook",
                "meta", "tesla", "nvidia"
            },
            "doc_types": {
                "10-k", "10k", "annual report", "10-q", "10q", "quarterly report",
                "8-k", "8k", "financial statement", "earnings report"
            },
            "financial_terms": {
                "risk factors", "revenue", "profit", "loss", "earnings",
                "income", "balance sheet", "cash flow", "dividend",
                "shareholders", "stock", "shares"
            }
        }

    def validate_rewrite(self, original: str, rewritten: str) -> bool:
        """
        Validate that the rewritten query preserves the intent and key elements
        of the original query.
        
        Args:
            original: Original query string
            rewritten: Rewritten query string
            
        Returns:
            bool: True if rewrite is valid, False otherwise
        """
        # Convert to lowercase for comparison
        original = original.lower()
        rewritten = rewritten.lower()
        
        # Check for meta-query patterns
        if self._is_meta_query(rewritten):
            logger.warning(
                "Rewrite rejected: Contains meta-query pattern. Original: '%s', Rewritten: '%s'",
                original, rewritten
            )
            return False
            
        # Extract and compare key elements
        original_elements = self._extract_elements(original)
        rewritten_elements = self._extract_elements(rewritten)
        
        # Validate element preservation
        if not self._validate_elements(original_elements, rewritten_elements):
            logger.warning(
                "Rewrite rejected: Key elements not preserved. Original: '%s', Rewritten: '%s'",
                original, rewritten
            )
            return False
            
        # Validate query length
        if len(rewritten) > len(original) * 2:
            logger.warning(
                "Rewrite rejected: Query too verbose. Original: '%s', Rewritten: '%s'",
                original, rewritten
            )
            return False
            
        logger.info(
            "Query rewrite validated successfully. Original: '%s', Rewritten: '%s'",
            original, rewritten
        )
        return True

    def _is_meta_query(self, query: str) -> bool:
        """Check if query matches any meta-query patterns."""
        return any(re.search(pattern, query, re.IGNORECASE) 
                  for pattern in self.meta_query_patterns)

    def _extract_elements(self, query: str) -> Dict[str, Set[str]]:
        """Extract key elements from query text."""
        elements = {
            "companies": set(),
            "doc_types": set(),
            "financial_terms": set()
        }
        
        for category, terms in self.key_entities.items():
            for term in terms:
                if term in query:
                    elements[category].add(term)
                    
        return elements

    def _validate_elements(
        self,
        original_elements: Dict[str, Set[str]],
        rewritten_elements: Dict[str, Set[str]]
    ) -> bool:
        """
        Validate that key elements are preserved between original
        and rewritten queries.
        """
        # All companies from original must be in rewrite
        if not original_elements["companies"].issubset(rewritten_elements["companies"]):
            return False
            
        # All document types from original must be in rewrite
        if not original_elements["doc_types"].issubset(rewritten_elements["doc_types"]):
            return False
            
        # At least one financial term from original must be in rewrite
        if (original_elements["financial_terms"] and 
            not original_elements["financial_terms"] & rewritten_elements["financial_terms"]):
            return False
            
        return True