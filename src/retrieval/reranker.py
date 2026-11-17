
from dataclasses import replace
from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class Reranker:

    def __init__(self):
        self.model = CrossEncoder(MODEL_NAME)

    def rerank(self, query, results, k=5):

        pairs = [
            (query, result.text)
            for result in results
        ]

        scores = self.model.predict(pairs)

        ranked_results = [
            replace(result, score=float(score), method="rerank")
            for result, score in zip(results, scores)
        ]

        ranked_results.sort(
            key=lambda result: result.score,
            reverse=True
        )

        return ranked_results[:k]
