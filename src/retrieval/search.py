"""Public entry point of the retrieval layer (used by the API and the agents).

Choose: the corpus ('fstbm' / 'ocr'), the role ('admin' / 'reader'),
the method ('vector' / 'bm25' / 'hybrid') and optionally the document type.
"""
from __future__ import annotations

from typing import Any, Callable

from .bm25 import Bm25Store
from .config import DEFAULT_TOP_K, SEARCH_MODES
from .hybrid import hybrid_search
from .loader import load_all
from .models import SearchResult


class Retriever:
    def __init__(self, vector_store=None, encode: Callable[[str], list[float]] | None = None):
        if vector_store is None:
            from .vector_store import ChromaStore

            vector_store = ChromaStore()
        if encode is None:
            from .query_encoder import QueryEncoder

            encode = QueryEncoder().encode
        self.vector_store = vector_store
        self.encode = encode
        self.corpora = load_all()
        self.bm25 = Bm25Store(self.corpora)

    def document_types(self, corpus: str) -> list[str]:
        """Document types available in a corpus (for a front-end dropdown)."""
        return sorted({c.document_type for c in self.corpora[corpus] if c.document_type})

    def search(
        self,
        query: str,
        corpus: str,
        role: str | None,
        k: int = DEFAULT_TOP_K,
        mode: str = "hybrid",
        document_type: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Empty query")
        if mode not in SEARCH_MODES:
            raise ValueError(f"mode must be one of {SEARCH_MODES}")
        all_filters = dict(filters or {})
        if document_type:
            all_filters["document_type"] = document_type
        all_filters = all_filters or None

        if mode == "bm25":
            return self.bm25.search(query, corpus, role, k, all_filters)
        if mode == "vector":
            return self.vector_store.search(self.encode(query), corpus, role, k, all_filters)
        return hybrid_search(
            query, corpus, role, k, self.vector_store, self.encode, self.bm25, all_filters
        )







































# """Public entry point of the retrieval layer (used by the API and the agents).

# Choose: the corpus ('fstbm' / 'ocr'), the role ('admin' / 'editeur' / 'lecteur'),
# the method ('vector' / 'bm25' / 'hybrid') and optionally the document type.
# """
# from __future__ import annotations

# from typing import Any, Callable

# from .bm25 import Bm25Store
# from .config import DEFAULT_TOP_K, SEARCH_MODES
# from .hybrid import hybrid_search
# from .loader import load_all
# from .models import SearchResult


# class Retriever:
#     def __init__(self, vector_store=None, encode: Callable[[str], list[float]] | None = None):
#         if vector_store is None:
#             from .vector_store import ChromaStore

#             vector_store = ChromaStore()
#         if encode is None:
#             from .query_encoder import QueryEncoder

#             encode = QueryEncoder().encode
#         self.vector_store = vector_store
#         self.encode = encode
#         self.corpora = load_all()
#         self.bm25 = Bm25Store(self.corpora)

#     def document_types(self, corpus: str) -> list[str]:
#         """Document types available in a corpus (for a front-end dropdown)."""
#         return sorted({c.document_type for c in self.corpora[corpus] if c.document_type})

#     def search(
#         self,
#         query: str,
#         corpus: str,
#         role: str | None,
#         k: int = DEFAULT_TOP_K,
#         mode: str = "hybrid",
#         document_type: str | None = None,
#         filters: dict[str, Any] | None = None,
#     ) -> list[SearchResult]:
#         if not query.strip():
#             raise ValueError("Empty query")
#         if mode not in SEARCH_MODES:
#             raise ValueError(f"mode must be one of {SEARCH_MODES}")
#         all_filters = dict(filters or {})
#         if document_type:
#             all_filters["document_type"] = document_type
#         all_filters = all_filters or None

#         if mode == "bm25":
#             return self.bm25.search(query, corpus, role, k, all_filters)
#         if mode == "vector":
#             return self.vector_store.search(self.encode(query), corpus, role, k, all_filters)
#         return hybrid_search(
#             query, corpus, role, k, self.vector_store, self.encode, self.bm25, all_filters
#         )
