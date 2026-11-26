"""Method 3 - hybrid search: gathers vector (Chroma) + BM25, then fuses with RRF (F-RET-02)."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from .bm25 import Bm25Store
from .config import CANDIDATES_PER_METHOD, RRF_K
from .models import SearchResult


def rrf_fuse(rankings: list[list[SearchResult]], k: int, rrf_k: int = RRF_K) -> list[SearchResult]:
    """score(chunk) = sum over lists of 1 / (rrf_k + rank), rank starting at 1."""
    scores: dict[str, float] = {}
    best: dict[str, SearchResult] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            best.setdefault(result.chunk_id, result)
    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:k]
    return [replace(best[cid], score=scores[cid], method="hybrid") for cid in ordered]


def hybrid_search(
    query: str,
    corpus: str,
    role: str | None,
    k: int,
    vector_store,
    encode: Callable[[str], list[float]],
    bm25_store: Bm25Store,
    filters: dict[str, Any] | None = None,
) -> list[SearchResult]:
    """Same role + same filters are applied to BOTH methods, then results are merged."""
    vector = vector_store.search(encode(query), corpus, role, CANDIDATES_PER_METHOD, filters)
    lexical = bm25_store.search(query, corpus, role, CANDIDATES_PER_METHOD, filters)
    return rrf_fuse([vector, lexical], k)
