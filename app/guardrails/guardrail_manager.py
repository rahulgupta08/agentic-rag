from app.guardrails.guardrail import GuardrailResult

class GuardrailManager:

    def __init__(
        self,
        domain_keywords=None,
        blocked_terms=None,
        llm=None
    ):
        self.domain_keywords = domain_keywords or []
        self.blocked_terms = blocked_terms or []
        self.llm = llm  # optional LLM-based policy check

    # ---------------------------
    # INPUT GUARDRAIL
    # ---------------------------
    def validate_input(self, query: str) -> GuardrailResult:

        query_lower = query.lower()

        #  Block obvious malicious content
        if any(term in query_lower for term in self.blocked_terms):
            return GuardrailResult(
                allowed=False,
                reason="Query contains restricted content."
            )

        #  Domain restriction
        if self.domain_keywords:
            if not any(keyword in query_lower for keyword in self.domain_keywords):
                return GuardrailResult(
                    allowed=False,
                    reason="Query is outside supported financial domain."
                )

        return GuardrailResult(allowed=True)

    # ---------------------------
    # OUTPUT GUARDRAIL
    # ---------------------------
    def validate_output(self, answer: str) -> GuardrailResult:

        # Example: Prevent disclosure patterns
        sensitive_patterns = ["ssn", "social security", "password"]

        answer_lower = answer.lower()

        if any(pattern in answer_lower for pattern in sensitive_patterns):
            return GuardrailResult(
                allowed=False,
                reason="Answer contains sensitive information."
            )

        return GuardrailResult(allowed=True)