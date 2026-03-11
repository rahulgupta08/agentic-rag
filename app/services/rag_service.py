from typing import Dict, List
import re
import asyncio



class RAGService:

    def __init__(self, retriever, llm, prompt_builder=None, query_expander=None, guardrail_manager=None):
        self.retriever = retriever
        self.llm = llm
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
    
    async def generate_from_context(self, query: str, context: str):
        """Generation only (used by agent)"""

        prompt = f"""
        Answer the question using the context below.

        Context:
        {context}

        Question:
        {query}
        """

        response = await self.llm.ainvoke(prompt)

        return response
            

    def generate(self, query: str, top_k: int = 5) -> Dict:


        # Step 0: Input guardrail check
        input_guardrail_validation = self.check_input_guardrail(query)
        if not input_guardrail_validation.allowed:
                return {
                            "answer": "This query cannot be processed.",
                            "reason": input_guardrail_validation.reason,
                            "confidence": 0.0,
                            "iterations": 0
                        }


        # Step 1 — Expand query (optional)
        if self.query_expander:
            expanded_query = self.query_expander.expand(query)
        else:
            expanded_query = query

        # Step 2: Retrieve documents
        documents = asyncio.run(self.retriever.retrieve(expanded_query, top_k=top_k))
        #documents = await self.retriever.retrieve(expanded_query, top_k=top_k)

        # Step 2: Build context
        context = self._build_context(documents)

        # Step 3: Build prompt
        #prompt = self._build_prompt(expanded_query, context)
        prompt = self.prompt_builder.build_prompt(expanded_query, context)

        # in case we want to build generation prompt instead of the default prompt

        #prompt = self.prompt_builder.build_generation_prompt(context=context,question=expanded_query)

        # Step 4: Call LLM
        answer =  self.llm.invoke(prompt)

        # In case we want to build reflection prompt instead of the default prompt
        # This is Self-Reflection Prompting
        # reflection_prompt = self.prompt_builder.build_reflection_prompt(
        #                         question=query,
        #                         answer=answer,
        #                         context=context
        #                     )
        # reflection_response = self.llm.invoke(reflection_prompt)
        # is_supported = "YES" in reflection_response.content.upper()
        # return {
        #     "answer": answer,
        #     "documents_used": documents,
        #     "supported": is_supported,
        #     "citations_valid": citations_valid
        # }

        # In case you need to validate citations in the answer against the retrieved documents
        #citations_valid = self._validate_citations(answer, documents)

        # Step 5: Output guardrail check
        output_guardrail_validation = self.check_output_guardrail(answer.content)
        if not output_guardrail_validation.allowed:
                return {
                            "answer": "The generated answer violates content policies and cannot be provided.",
                            "reason": output_guardrail_validation.reason,
                            "confidence": 0.0,
                            "iterations": 0
                        }

        return {
            "query": expanded_query,
            "answer": answer.content,
            "documents_used": documents
        }

    def _build_context(self, documents: List[Dict]) -> str:
        context_blocks = []

        for doc in documents:
                block = f"""
                        [Document ID: {doc['id']}]
                        {doc['text']}
                        """
                context_blocks.append(block)

        return "\n\n".join(context_blocks)
    
    def _validate_citations(self, answer: str, documents: list) -> bool:

        cited_ids = re.findall(r"\[Document ID:\s*(.*?)\]", answer)

        valid_ids = {doc["id"] for doc in documents}

        return all(cid in valid_ids for cid in cited_ids)

    #use it for iterative retrieval instead of a simple generate funtion
    def generate_iterative_rag(self, query: str, top_k: int = 5, max_iterations: int = 2):

        current_query = query
        iteration = 0
        
        # Step 0: Input guardrail check
        input_guardrail_validation = self.check_input_guardrail(query)
        if not input_guardrail_validation.allowed:
                return {
                            "answer": "This query cannot be processed.",
                            "reason": input_guardrail_validation.reason,
                            "confidence": 0.0,
                            "iterations": 0
                        }

        while iteration < max_iterations:

            # Step 1: Expand query (if enabled)
            if self.query_expander:
                expanded_query = self.query_expander.expand(current_query)
            else:
                expanded_query = current_query

            # Step 2: Retrieve
            documents = asyncio.run(self.retriever.retrieve(expanded_query, top_k=top_k))

            context = self._build_context(documents)

            # Step 3: Generate
            prompt = self.prompt_builder.build_generation_prompt(
                context=context,
                question=expanded_query
            )

            response = self.llm.invoke(prompt)
            answer = response.content

            # Step 4: Validate citations
            citations_valid = self._validate_citations(answer, documents)

            # Step 5: Reflect
            reflection_prompt = self.prompt_builder.build_reflection_prompt(
                question=expanded_query,
                answer=answer,
                context=context
            )

            reflection_response = self.llm.invoke(reflection_prompt)
            supported = "YES" in reflection_response.content.upper()

            # Step 6 Instead of just checking for yes/no, use this to compute confidence scpre for the generated answer based on the context and the question.
            # Either use step 5 if not step 5 then use this confidence score.
            # confidence = self._compute_confidence(query, answer, context)

            # confidence_threshold = 0.75

            # if confidence >= confidence_threshold and citations_valid:
            #     return {
            #         "answer": answer,
            #         "documents_used": documents,
            #         "confidence": confidence,
            #         "iterations": iteration + 1
            #     }

            # If valid → return
            if supported and citations_valid:
                return {
                    "answer": answer,
                    "documents_used": documents,
                    "supported": True,
                    "iterations": iteration + 1
                }

            # Else refine query
            refined_prompt = self.prompt_builder.refine_query_for_iterative_retrieval(expanded_query, answer)
            answer = self.llm.invoke(refined_prompt)

            return answer.content.strip()
        iteration += 1

        # Return best attempt after max retries
        return {
            "answer": answer,
            "documents_used": documents,
            "supported": False,
            "iterations": iteration
        }
    
    # use this to compute confidence score for the generated answer based on the context and the question.
    def _compute_confidence(self, query, answer, context):

            confidence_prompt = self.prompt_builder.build_confidence_prompt(
                question=query,
                answer=answer,
                context=context
            )

            response = self.llm.invoke(confidence_prompt)

            try:
                score = float(response.content.strip())
                return max(0.0, min(score, 1.0))
            except:
                return 0.0
            
    