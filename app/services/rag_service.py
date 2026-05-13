from typing import Dict, List
import re
import asyncio
import logging
logger = logging.getLogger(__name__)


class RAGService:

    def __init__(
        self,
        retriever,
        llm,
        query_classifier,
        query_router,
        prompt_builder=None,
        query_expander=None,
        guardrail_manager=None
    ):
        self.retriever = retriever
        self.llm = llm
        self.query_classifier = query_classifier
        self.query_router = query_router
        self.prompt_builder = prompt_builder
        self.query_expander = query_expander
        self.guardrail_manager = guardrail_manager

    def check_input_guardrail(self, query: str) -> Dict:
        if self.guardrail_manager:
            input_validation = self.guardrail_manager.validate_input(query)
            return input_validation
        class DefaultValidation:
            allowed = True
            reason = None
        return DefaultValidation()
        
    def check_output_guardrail(self, answer: str) -> Dict:
        if self.guardrail_manager:
            output_validation = self.guardrail_manager.validate_output(answer)
            return output_validation
        class DefaultValidation:
            allowed = True
            reason = None
        return DefaultValidation()

    def generate(self, query: str, top_k: int = 5) -> Dict:
        """Synchronous generation for RAG mode"""
        logger.info("Processing query: %s", query)

        # Step 0: Input guardrail check
        input_guardrail_validation = self.check_input_guardrail(query)
        if not input_guardrail_validation.allowed:
            return {
                "answer": "This query cannot be processed.",
                "reason": input_guardrail_validation.reason,
                "confidence": 0.0,
                "iterations": 0
            }

        # Step 1: Early query classification and routing
        route_result = self.query_router.route_query(query)
        logger.info(
            "Query routed to %s (skip_retrieval=%s, skip_rerank=%s): %s",
            route_result.route,
            route_result.skip_retrieval,
            route_result.skip_rerank,
            route_result.reason
        )

        # Handle fast path and direct responses
        if route_result.route in ("fast_path", "direct_response"):
            template = self.query_router.get_response_template(route_result)
            return {
                "answer": template,
                "confidence": 1.0,
                "iterations": 0,
                "route": route_result.route
            }

        # Step 2: Query expansion (optional)
        if self.query_expander:
            expanded_query = self.query_expander.transform(query)[0]
            logger.info("Expanded query: %s", expanded_query)
        else:
            expanded_query = query

        # Step 3: Retrieve and process documents
        if not route_result.skip_retrieval:
            logger.info("Retrieving documents for query: %s (top_k=%d)", expanded_query, top_k)
            documents = self.retriever.retrieve(
                expanded_query,
                top_k=top_k,
                skip_rerank=route_result.skip_rerank
            )
            logger.info("Retrieved %d documents", len(documents))
            context = self._build_context(documents)
        else:
            logger.info("Skipping retrieval based on query classification")
            context = ""

        # Step 4: Build prompt
        logger.info("Building prompt with context length: %d chars", len(context))
        prompt = self.prompt_builder.build_prompt(expanded_query, context)
        logger.debug("Built prompt length: %d chars", len(prompt))

        # Step 5: Call LLM
        logger.info("Invoking LLM for generation")
        answer = self.llm.invoke(prompt)
        logger.info("Generated answer length: %d chars", len(answer))

        # Step 6: Output guardrail check
        output_guardrail_validation = self.check_output_guardrail(answer)
        if not output_guardrail_validation.allowed:
            return {
                "answer": "The generated response cannot be shown.",
                "reason": output_guardrail_validation.reason,
                "confidence": 0.0,
                "iterations": 0
            }

        return {
            "answer": answer,
            "confidence": self._calculate_confidence(answer),
            "iterations": 1,
            "route": route_result.route
        }

    async def generate_from_context(self, query: str, context: str):
        """Asynchronous generation for Agent mode"""
        prompt = f"""
        Answer the question using the context below.

        Context:
        {context}

        Question:
        {query}
        """

        response = await self.llm.ainvoke(prompt)
        return response

    async def agenerate(self, query: str, top_k: int = 5) -> Dict:
        """Asynchronous generation for Agent mode"""
        logger.info("Processing query (async): %s", query)

        # Step 0: Input guardrail check
        input_guardrail_validation = self.check_input_guardrail(query)
        if not input_guardrail_validation.allowed:
            return {
                "answer": "This query cannot be processed.",
                "reason": input_guardrail_validation.reason,
                "confidence": 0.0,
                "iterations": 0
            }

        # Step 1: Early query classification and routing
        route_result = self.query_router.route_query(query)
        logger.info(
            "Query routed to %s (skip_retrieval=%s, skip_rerank=%s): %s",
            route_result.route,
            route_result.skip_retrieval,
            route_result.skip_rerank,
            route_result.reason
        )

        # Handle fast path and direct responses
        if route_result.route in ("fast_path", "direct_response"):
            template = self.query_router.get_response_template(route_result)
            return {
                "answer": template,
                "confidence": 1.0,
                "iterations": 0,
                "route": route_result.route
            }

        # Step 2: Query expansion (optional)
        if self.query_expander:
            expanded_queries = await self.query_expander.transform_async(query)
            expanded_query = expanded_queries[0]
            logger.info("Expanded query: %s", expanded_query)
        else:
            expanded_query = query

        # Step 3: Retrieve and process documents
        if not route_result.skip_retrieval:
            logger.info("Retrieving documents for query: %s (top_k=%d)", expanded_query, top_k)
            documents = await self.retriever.aretrieve(
                expanded_query,
                top_k=top_k,
                skip_rerank=route_result.skip_rerank
            )
            logger.info("Retrieved %d documents", len(documents))
            context = self._build_context(documents)
        else:
            logger.info("Skipping retrieval based on query classification")
            context = ""

        # Step 4: Build prompt
        logger.info("Building prompt with context length: %d chars", len(context))
        prompt = self.prompt_builder.build_prompt(expanded_query, context)
        logger.debug("Built prompt length: %d chars", len(prompt))

        # Step 5: Call LLM
        logger.info("Invoking LLM for generation")
        answer = await self.llm.ainvoke(prompt)
        logger.info("Generated answer length: %d chars", len(answer))

        # Step 6: Output guardrail check
        output_guardrail_validation = self.check_output_guardrail(answer)
        if not output_guardrail_validation.allowed:
            return {
                "answer": "The generated response cannot be shown.",
                "reason": output_guardrail_validation.reason,
                "confidence": 0.0,
                "iterations": 0
            }

        return {
            "answer": answer,
            "confidence": self._calculate_confidence(answer),
            "iterations": 1,
            "route": route_result.route
        }

    def _build_context(self, documents: List[Dict]) -> str:
        """Helper method to build context from retrieved documents"""
        if not documents:
            logger.warning("No documents received for context building")
            return ""

        context_parts = []
        for i, doc in enumerate(documents):
            # Try both "text" and "content" keys for backward compatibility
            content = doc.get("text", doc.get("content", "")).strip()
            
            if not content:
                logger.warning(f"Document {i} has no content (keys present: {list(doc.keys())})")
                continue
                
            context_parts.append(content)

        context = "\n\n".join(context_parts)
        
        logger.info(
            "Built context from %d/%d documents (total length: %d chars)",
            len(context_parts),
            len(documents),
            len(context)
        )
        
        if context:
            logger.debug(
                "Context preview (first 200 chars): %s...",
                context[:200]
            )
        
        return context

    def _calculate_confidence(self, answer: str) -> float:
        """Helper method to calculate confidence score"""
        # Simple heuristic based on answer length and structure
        if len(answer) < 10:
            return 0.3
        elif "I don't know" in answer or "cannot" in answer:
            return 0.4
        else:
            return 0.8  # Default confidence for complete-looking answers