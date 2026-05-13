import logging
from typing import List
from app.query.query_validator import QueryValidator

logger = logging.getLogger(__name__)

QUERY_REWRITE_PROMPT = """
Your task is to refine a search query while preserving its original intent.

Guidelines:
1. NEVER change the query into a meta-query or query generation task
2. Maintain the original:
   - Information seeking intent
   - Entity references (companies, document types)
   - Financial terms and concepts
3. Focus on enhancing searchability while keeping meaning intact
4. Keep the rewrite concise and focused

Examples:
Original: "What risks does Apple talk about?"
Good: "Identify risk factors disclosed by Apple"
Bad: "Generate queries about Apple's risks"

Original: "Show me Apple revenue in 10K"
Good: "Find Apple's revenue information in their 10-K filing"
Bad: "List different ways to search Apple's 10-K for revenue"

Original Query: {query}

Rewrite this query to improve search effectiveness while preserving its core intent.
Respond with ONLY the rewritten query, no explanations or additional text.
"""

class LLMQueryRewriter:
    """
    Rewrites queries using LLM while preserving original intent
    and key information elements.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.validator = QueryValidator()
        
    def transform(self, query: str) -> List[str]:
        """
        Transform a query while preserving its intent.
        Returns a list containing the single rewritten query.
        """
        logger.info("Original query: %s", query)
        
        # Get rewritten query from LLM
        rewritten = self._rewrite_query(query)
        
        # Validate the rewrite
        if not self.validator.validate_rewrite(query, rewritten):
            logger.warning(
                "Query rewrite failed validation, using original query: %s",
                query
            )
            return [query]
            
        logger.info("Rewritten query: %s", rewritten)
        return [rewritten]
        
    async def transform_async(self, query: str) -> List[str]:
        """
        Async version of transform.
        Returns a list containing the single rewritten query.
        """
        logger.info("Original query (async): %s", query)
        
        # Get rewritten query from LLM
        rewritten = await self._rewrite_query_async(query)
        
        # Validate the rewrite
        if not self.validator.validate_rewrite(query, rewritten):
            logger.warning(
                "Query rewrite failed validation, using original query: %s",
                query
            )
            return [query]
            
        logger.info("Rewritten query (async): %s", rewritten)
        return [rewritten]
        
    def _rewrite_query(self, query: str) -> str:
        """Synchronous query rewriting using LLM."""
        prompt = QUERY_REWRITE_PROMPT.format(query=query)
        
        try:
            rewritten = self.llm.invoke(prompt).strip()
            return rewritten
        except Exception as e:
            logger.error("Query rewrite failed: %s", str(e))
            return query
            
    async def _rewrite_query_async(self, query: str) -> str:
        """Asynchronous query rewriting using LLM."""
        prompt = QUERY_REWRITE_PROMPT.format(query=query)
        
        try:
            rewritten = (await self.llm.ainvoke(prompt)).strip()
            return rewritten
        except Exception as e:
            logger.error("Query rewrite failed (async): %s", str(e))
            return query