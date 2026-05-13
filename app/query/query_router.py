import logging
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RouteConfig:
    """Configuration for query routing decisions."""
    confidence_threshold: float = 0.5
    force_fast_path_types: set = None
    
    def __post_init__(self):
        if self.force_fast_path_types is None:
            self.force_fast_path_types = {"personal", "unanswerable"}

@dataclass
class RouteResult:
    """Result of query routing decision."""
    route: str
    reason: str
    skip_retrieval: bool
    skip_rerank: bool

class QueryRouter:
    """
    Routes queries to appropriate processing paths based on classification.
    Determines whether to use fast path, skip reranking, etc.
    """

    def __init__(self, classifier, config: Optional[RouteConfig] = None):
        self.classifier = classifier
        self.config = config or RouteConfig()

    def route_query(self, query: str) -> RouteResult:
        """
        Determine processing route for a query.
        
        Args:
            query: The input query string
            
        Returns:
            RouteResult containing routing decision and flags
        """
        # Get query classification
        classification = self.classifier.classify_query(query)
        
        # Force fast path for certain query types
        if classification["classification"] in self.config.force_fast_path_types:
            return self._create_fast_path_result(classification["classification"])
            
        # Route based on confidence
        if classification["confidence"] < self.config.confidence_threshold:
            return RouteResult(
                route="direct_response",
                reason=f"Low confidence ({classification['confidence']:.2f})",
                skip_retrieval=True,
                skip_rerank=True
            )
            
        # Standard processing with optional reranking
        return self._create_standard_result(classification)

    def _create_fast_path_result(self, classification: str) -> RouteResult:
        """Create result for fast path routing."""
        return RouteResult(
            route="fast_path",
            reason=f"Query classified as {classification}",
            skip_retrieval=True,
            skip_rerank=True
        )

    def _create_standard_result(self, classification: Dict) -> RouteResult:
        """Create result for standard processing path."""
        # Determine if reranking should be skipped
        skip_rerank = classification["confidence"] < 0.7
        
        return RouteResult(
            route="full_pipeline",
            reason="Standard processing",
            skip_retrieval=False,
            skip_rerank=skip_rerank
        )

    def get_response_template(self, route_result: RouteResult) -> str:
        """Get appropriate response template for route."""
        if route_result.route == "fast_path":
            if "personal" in route_result.reason:
                return "I cannot answer personal questions about you or your context."
            else:
                return "I cannot answer this type of question."
                
        elif route_result.route == "direct_response":
            return "I don't have enough confidence to answer this question accurately."
            
        return None  # No template for full pipeline