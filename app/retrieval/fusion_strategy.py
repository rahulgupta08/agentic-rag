class WeightedFusion:

    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha

    def merge(self, dense_docs, bm25_docs, top_k):

        doc_map = {}

        for d in dense_docs:
            doc_map[d["text"]] = {
                **d,
                "dense_score": d["score"],
                "bm25_score": 0
            }

        for d in bm25_docs:
            if d["text"] in doc_map:
                doc_map[d["text"]]["bm25_score"] = d["score"]
            else:
                doc_map[d["text"]] = {
                    **d,
                    "dense_score": 0,
                    "bm25_score": d["score"]
                }

        docs = list(doc_map.values())

        docs = self._normalize(docs, "dense_score")
        docs = self._normalize(docs, "bm25_score")

        for d in docs:
            d["score"] = (
                self.alpha * d["dense_score_norm"] +
                (1 - self.alpha) * d["bm25_score_norm"]
            )

        return sorted(docs, key=lambda x: x["score"], reverse=True)[:top_k]

    def _normalize(self, docs, key):
        scores = [d[key] for d in docs]
        min_s, max_s = min(scores), max(scores)

        for d in docs:
            if max_s - min_s == 0:
                d[f"{key}_norm"] = 0
            else:
                d[f"{key}_norm"] = (d[key] - min_s) / (max_s - min_s)

        return docs