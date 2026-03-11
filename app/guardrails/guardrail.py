class GuardrailResult:
    def __init__(self, allowed: bool, reason: str = None):
        self.allowed = allowed
        self.reason = reason