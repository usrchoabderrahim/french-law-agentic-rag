"""Method 2 - lexical search (BM25), implemented without external dependency.

The index is built per (corpus, role) from the chunks that role may see, so the
role filter is applied BEFORE any scoring (F-SEC-04).
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Callable

from .loader import IndexedChunk
from .models import SearchResult
from .vector_store import check_role

STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "l", "et", "ou", "en",
    "au", "aux", "a", "est", "sont", "ce", "ces", "cette", "que", "qui", "quoi",
    "dans", "par", "pour", "sur", "sous", "il", "elle", "ils", "elles", "se", "sa",
    "son", "ses", "ne", "pas", "plus", "the", "of", "and", "to", "in", "is", "are",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, remove accents, split into words, drop stopwords."""
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch)).lower()
    tokens = re.findall(r"\w+", folded)
    return [t for t in tokens if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


class BM25Index:
    def __init__(self, chunks: list[IndexedChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for idx, chunk in enumerate(chunks):
            counts = Counter(tokenize(chunk.text))
            self.doc_len.append(sum(counts.values()))
            for term, freq in counts.items():
                self.postings.setdefault(term, []).append((idx, freq))
        n = len(chunks)
        self.avg_len = (sum(self.doc_len) / n) if n else 0.0
        self.idf = {
            term: math.log((n - len(p) + 0.5) / (len(p) + 0.5) + 1.0)
            for term, p in self.postings.items()
        }

    def search(
        self, query: str, k: int, accept: Callable[[IndexedChunk], bool] | None = None
    ) -> list[tuple[IndexedChunk, float]]:
        scores: dict[int, float] = {}
        for term in set(tokenize(query)):
            for idx, freq in self.postings.get(term, []):
                if accept is not None and not accept(self.chunks[idx]):
                    continue  # metadata filter applied before scoring
                norm = 1 - self.b + self.b * self.doc_len[idx] / self.avg_len
                score = self.idf[term] * freq * (self.k1 + 1) / (freq + self.k1 * norm)
                scores[idx] = scores.get(idx, 0.0) + score
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        return [(self.chunks[idx], score) for idx, score in ranked]


class Bm25Store:
    """Holds one BM25 index per (corpus, role), built lazily."""

    def __init__(self, corpora: dict[str, list[IndexedChunk]]):
        self.corpora = corpora
        self._indexes: dict[tuple[str, str], BM25Index] = {}

    def search(
        self, query: str, corpus: str, role: str | None, k: int, filters: dict | None = None
    ) -> list[SearchResult]:
        role = check_role(role)
        if corpus not in self.corpora:
            raise ValueError(f"Unknown corpus '{corpus}'. Choose from {list(self.corpora)}")
        key = (corpus, role)
        if key not in self._indexes:
            allowed = [c for c in self.corpora[corpus] if role in c.allowed_roles]
            self._indexes[key] = BM25Index(allowed)
        accept = None
        if filters:
            accept = lambda c: all(getattr(c, f, None) == v for f, v in filters.items())  # noqa: E731
        results = []
        for chunk, score in self._indexes[key].search(query, k, accept):
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=score,
                    method="bm25",
                    corpus=corpus,
                    title=chunk.title,
                    source_url=chunk.source_url,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section=chunk.section,
                    article_number=chunk.article_number,
                    document_type=chunk.document_type,
                    publication_year=chunk.publication_year,
                )
            )
        return results
