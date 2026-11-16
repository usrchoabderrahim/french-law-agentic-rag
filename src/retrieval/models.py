"""Data objects shared by all search methods."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float  # meaning depends on the method (cosine, BM25 or RRF)
    method: str   # "vector", "bm25" or "hybrid"
    corpus: str
    title: str
    source_url: str | None
    page_start: int | None
    page_end: int | None
    section: str | None
    article_number: str | None
    document_type: str | None
    publication_year: int | None
