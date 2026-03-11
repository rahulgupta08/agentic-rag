class SimilarityThresholdFilter:

    def __init__(self, threshold=0.7):
        self.threshold = threshold

    def filter(self, docs):
        return [
            d for d in docs
            if d["score"] is not None and d["score"] >= self.threshold
        ]