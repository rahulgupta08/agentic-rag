class ThresholdFilter:

    def __init__(self, threshold: float):
        self.threshold = threshold

    def process(self, docs):
        return [d for d in docs if d["score"] >= self.threshold]