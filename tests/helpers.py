"""Shared helpers for the tests (no Chroma needed)."""
import numpy as np

from src.retrieval.loader import IndexedChunk, load_all
from src.retrieval.models import SearchResult
from src.retrieval.vector_store import build_where


def make_chunk(cid, text, roles, doc_type="guide"):
    return IndexedChunk(cid, "fstbm", text, np.zeros(1024), "d", "t", "s", None, doc_type,
                        "fr", None, 1, 1, None, None, roles)


def make_result(cid):
    return SearchResult(cid, "x", 0.0, "vector", "fstbm", "t", None, 1, 1, None, None, None, None)


class FakeVectorStore:
    """Same API as ChromaStore.search, brute-force numpy, same role/filter rules."""

    def __init__(self):
        self.corpora = load_all()

    def search(self, query_vector, corpus, role, k, filters=None):
        build_where(role, filters)  # same role check as Chroma
        data = [c for c in self.corpora[corpus] if role in c.allowed_roles]
        if filters:
            data = [c for c in data if all(getattr(c, f, None) == v for f, v in filters.items())]
        sims = np.array([c.embedding for c in data]) @ np.array(query_vector)
        return [make_result(data[i].chunk_id) for i in np.argsort(-sims)[:k]]









































# """Shared helpers for the tests (no Chroma needed)."""
# import numpy as np

# from src.retrieval.loader import IndexedChunk, load_all
# from src.retrieval.models import SearchResult
# from src.retrieval.vector_store import build_where


# def make_chunk(cid, text, roles, doc_type="guide"):
#     return IndexedChunk(cid, "fstbm", text, np.zeros(1024), "d", "t", "s", None, doc_type,
#                         "fr", None, 1, 1, None, None, roles)


# def make_result(cid):
#     return SearchResult(cid, "x", 0.0, "vector", "fstbm", "t", None, 1, 1, None, None, None, None)


# class FakeVectorStore:
#     """Same API as ChromaStore.search, brute-force numpy, same role/filter rules."""

#     def __init__(self):
#         self.corpora = load_all()

#     def search(self, query_vector, corpus, role, k, filters=None):
#         build_where(role, filters)  # same role check as Chroma
#         data = [c for c in self.corpora[corpus] if (role if role != "user" else "lecteur") in c.allowed_roles]
#         if filters:
#             data = [c for c in data if all(getattr(c, f, None) == v for f, v in filters.items())]
#         sims = np.array([c.embedding for c in data]) @ np.array(query_vector)
#         return [make_result(data[i].chunk_id) for i in np.argsort(-sims)[:k]]
