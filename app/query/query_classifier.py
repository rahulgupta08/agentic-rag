import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class QueryClassifier:
    """
    Classifies queries to determine appropriate processing path.
    Handles detection of personal queries, unanswerable queries,
    and provides confidence scores for query routing.
    """

    def __init__(self):
        # Personal query patterns
        self.personal_patterns = [
            r'\b(my|me|i|we|our)\b',
            r'\b(mine|ours)\b',
            r'\b(myself|ourselves)\b'
        ]
        
        # Question patterns that are likely unanswerable
        self.unanswerable_patterns = [
            r'\b(where am i|who am i|what is my)\b',
            r'\b(what time is it|what day is)\b',
            r'\b(what is your|who are you)\b'
        ]

    def classify_query(self, query: str) -> Dict:
        """
        Classify a query to determine its properties and routing.
        
        Args:
            query: The input query string
            
        Returns:
            Dict containing classification results:
            {
                "is_personal": bool,
                "is_answerable": bool,
                "confidence": float,
                "classification": str
            }
        """
        is_personal = self._is_personal_query(query)
        is_answerable = self._is_answerable(query)
        confidence = self._get_confidence(query)
        
        classification = self._determine_classification(
            is_personal, 
            is_answerable,
            confidence
        )
        
        result = {
            "is_personal": is_personal,
            "is_answerable": is_answerable,
            "confidence": confidence,
            "classification": classification
        }
        
        logger.info(
            "Query classification: %s -> %s (conf=%.2f)", 
            query, 
            classification,
            confidence
        )
        
        return result

    def _is_personal_query(self, query: str) -> bool:
        """Check if query contains personal pronouns or context."""
        query = query.lower()
        return any(re.search(pattern, query) for pattern in self.personal_patterns)

    def _is_answerable(self, query: str) -> bool:
        """Determine if query can be answered from document collection."""
        query = query.lower()
        
        # Check for explicitly unanswerable patterns
        if any(re.search(pattern, query) for pattern in self.unanswerable_patterns):
            return False
            
        # Check for personal queries (subset of unanswerable)
        if self._is_personal_query(query):
            return False
            
        return True

    def _get_confidence(self, query: str) -> float:
        """
        Calculate confidence score for query answerability.
        
        Returns:
            Float between 0-1 indicating confidence
        """
        if not query.strip():
            return 0.0
            
        # Lower confidence for very short queries
        if len(query.split()) < 3:
            return 0.4
            
        # Lower confidence for personal queries
        if self._is_personal_query(query):
            return 0.3
            
        # Lower confidence for likely unanswerable queries
        if not self._is_answerable(query):
            return 0.2
            
        # Default confidence for standard queries
        return 0.8

    def _determine_classification(
        self,
        is_personal: bool,
        is_answerable: bool,
        confidence: float
    ) -> str:
        """Determine final query classification."""
        if is_personal:
            return "personal"
        elif not is_answerable:
            return "unanswerable"
        elif confidence < 0.5:
            return "low_confidence"
        else:
            return "standard"