"""Full search pipeline: search (vector | bm25 | hybrid) -> optional re-ranking."""
from __future__ import annotations

from typing import Any

from .config import CANDIDATES_PER_METHOD, DEFAULT_TOP_K
from .models import SearchResult
from .reranker import Reranker
from .search import Retriever


class LegalSearchPipeline:
    def __init__(self, retriever: Retriever | None = None, reranker: Reranker | None = None):
        if retriever is None:
            from .vector_store import ChromaStore

            store = ChromaStore()
            store.build_all()  # skipped automatically if already indexed
            retriever = Retriever(vector_store=store)
        self.retriever = retriever
        self.reranker = reranker or Reranker()

    def search(
        self,
        query: str,
        corpus: str,
        role: str | None,
        k: int = DEFAULT_TOP_K,
        mode: str = "hybrid",
        document_type: str | None = None,
        filters: dict[str, Any] | None = None,
        rerank: bool = True,
        candidates: int = CANDIDATES_PER_METHOD,
    ) -> list[SearchResult]:
        """1) retrieve `candidates` chunks with the chosen mode, 2) re-rank, keep top k."""
        pool = max(candidates, k) if rerank else k
        results = self.retriever.search(
            query, corpus, role, k=pool, mode=mode, document_type=document_type, filters=filters
        )
        if not rerank or not results:
            return results[:k]
        return self.reranker.rerank(query, results, k=k)