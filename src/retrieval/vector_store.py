"""Method 1 - semantic search with ChromaDB (one collection per corpus)."""
from __future__ import annotations

from typing import Any

from .config import CHROMA_DIR, COLLECTIONS, CORPORA, ROLES
from .loader import IndexedChunk, load_corpus
from .models import SearchResult

VALID_ROLES = set(ROLES)  # {"admin", "reader"}


def check_role(role: str | None) -> str:
    """T-14: a request without a valid role is rejected."""
    if role not in VALID_ROLES:
        raise PermissionError(f"Role required and must be one of {sorted(VALID_ROLES)}")
    return role


def to_metadata(chunk: IndexedChunk) -> dict[str, Any]:
    """Chroma only accepts scalar metadata (no list, no None): roles become booleans."""
    meta: dict[str, Any] = {
        "corpus": chunk.corpus,
        "document_id": chunk.document_id,
        "title": chunk.title,
        "source_name": chunk.source_name,
        "source_url": chunk.source_url,
        "document_type": chunk.document_type,
        "language": chunk.language,
        "publication_year": chunk.publication_year,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "section": chunk.section,
        "article_number": chunk.article_number,
    }
    for role in VALID_ROLES:
        meta[f"role_{role}"] = role in chunk.allowed_roles
    return {k: v for k, v in meta.items() if v is not None}


def build_where(role: str | None, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Role filter (mandatory) + optional metadata filters (F-RET-04)."""
    clauses: list[dict[str, Any]] = [{f"role_{check_role(role)}": True}]
    clauses += [{key: value} for key, value in (filters or {}).items()]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class ChromaStore:
    def __init__(self, path=CHROMA_DIR):
        import chromadb

        self.client = chromadb.PersistentClient(path=str(path))

    def _collection(self, corpus: str):
        if corpus not in COLLECTIONS:
            raise ValueError(f"Unknown corpus '{corpus}'. Choose from {list(COLLECTIONS)}")
        return self.client.get_or_create_collection(
            name=COLLECTIONS[corpus],
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,  # vectors are precomputed (bge-m3)
        )

    def build(self, corpus: str, force: bool = False, batch_size: int = 256) -> int:
        """Index one corpus in Chroma. Skipped if already indexed."""
        chunks = load_corpus(corpus)
        collection = self._collection(corpus)
        if collection.count() == len(chunks) and not force:
            return collection.count()
        self.client.delete_collection(COLLECTIONS[corpus])
        collection = self._collection(corpus)
        for i in range(0, len(chunks), batch_size):
            part = chunks[i : i + batch_size]
            collection.add(
                ids=[c.chunk_id for c in part],
                embeddings=[c.embedding.tolist() for c in part],
                documents=[c.text for c in part],
                metadatas=[to_metadata(c) for c in part],
            )
        return collection.count()

    def build_all(self, force: bool = False) -> dict[str, int]:
        return {name: self.build(name, force=force) for name in CORPORA}

    def search(
        self,
        query_vector: list[float],
        corpus: str,
        role: str | None,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        where = build_where(role, filters)  # raises if no role
        collection = self._collection(corpus)
        total = collection.count()
        if total == 0:
            return []
        raw = collection.query(
            query_embeddings=[query_vector],
            n_results=min(k, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        results = []
        for cid, doc, meta, dist in zip(
            raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
        ):
            results.append(
                SearchResult(
                    chunk_id=cid,
                    text=doc,
                    score=1.0 - float(dist),  # cosine similarity
                    method="vector",
                    corpus=corpus,
                    title=meta.get("title", ""),
                    source_url=meta.get("source_url"),
                    page_start=meta.get("page_start"),
                    page_end=meta.get("page_end"),
                    section=meta.get("section"),
                    article_number=meta.get("article_number"),
                    document_type=meta.get("document_type"),
                    publication_year=meta.get("publication_year"),
                )
            )
        return results









































# """Method 1 - semantic search with ChromaDB (one collection per corpus)."""
# from __future__ import annotations

# from typing import Any

# from .config import CHROMA_DIR, COLLECTIONS, CORPORA, ROLE_ALIASES
# from .loader import IndexedChunk, load_corpus
# from .models import SearchResult

# VALID_ROLES = {"admin", "editeur", "lecteur"}


# def check_role(role: str | None) -> str:
#     """T-14: a request without a valid role is rejected. 'user'/'reader' -> 'lecteur'."""
#     role = ROLE_ALIASES.get(role, role) if role else role
#     if role not in VALID_ROLES:
#         raise PermissionError(f"Role required and must be one of {sorted(VALID_ROLES)}")
#     return role


# def to_metadata(chunk: IndexedChunk) -> dict[str, Any]:
#     """Chroma only accepts scalar metadata (no list, no None): roles become booleans."""
#     meta: dict[str, Any] = {
#         "corpus": chunk.corpus,
#         "document_id": chunk.document_id,
#         "title": chunk.title,
#         "source_name": chunk.source_name,
#         "source_url": chunk.source_url,
#         "document_type": chunk.document_type,
#         "language": chunk.language,
#         "publication_year": chunk.publication_year,
#         "page_start": chunk.page_start,
#         "page_end": chunk.page_end,
#         "section": chunk.section,
#         "article_number": chunk.article_number,
#     }
#     for role in VALID_ROLES:
#         meta[f"role_{role}"] = role in chunk.allowed_roles
#     return {k: v for k, v in meta.items() if v is not None}


# def build_where(role: str | None, filters: dict[str, Any] | None = None) -> dict[str, Any]:
#     """Role filter (mandatory) + optional metadata filters (F-RET-04)."""
#     clauses: list[dict[str, Any]] = [{f"role_{check_role(role)}": True}]
#     clauses += [{key: value} for key, value in (filters or {}).items()]
#     return clauses[0] if len(clauses) == 1 else {"$and": clauses}


# class ChromaStore:
#     def __init__(self, path=CHROMA_DIR):
#         import chromadb

#         self.client = chromadb.PersistentClient(path=str(path))

#     def _collection(self, corpus: str):
#         if corpus not in COLLECTIONS:
#             raise ValueError(f"Unknown corpus '{corpus}'. Choose from {list(COLLECTIONS)}")
#         return self.client.get_or_create_collection(
#             name=COLLECTIONS[corpus],
#             metadata={"hnsw:space": "cosine"},
#             embedding_function=None,  # vectors are precomputed (bge-m3)
#         )

#     def build(self, corpus: str, force: bool = False, batch_size: int = 256) -> int:
#         """Index one corpus in Chroma. Skipped if already indexed."""
#         chunks = load_corpus(corpus)
#         collection = self._collection(corpus)
#         if collection.count() == len(chunks) and not force:
#             return collection.count()
#         self.client.delete_collection(COLLECTIONS[corpus])
#         collection = self._collection(corpus)
#         for i in range(0, len(chunks), batch_size):
#             part = chunks[i : i + batch_size]
#             collection.add(
#                 ids=[c.chunk_id for c in part],
#                 embeddings=[c.embedding.tolist() for c in part],
#                 documents=[c.text for c in part],
#                 metadatas=[to_metadata(c) for c in part],
#             )
#         return collection.count()

#     def build_all(self, force: bool = False) -> dict[str, int]:
#         return {name: self.build(name, force=force) for name in CORPORA}

#     def search(
#         self,
#         query_vector: list[float],
#         corpus: str,
#         role: str | None,
#         k: int,
#         filters: dict[str, Any] | None = None,
#     ) -> list[SearchResult]:
#         where = build_where(role, filters)  # raises if no role
#         collection = self._collection(corpus)
#         total = collection.count()
#         if total == 0:
#             return []
#         raw = collection.query(
#             query_embeddings=[query_vector],
#             n_results=min(k, total),
#             where=where,
#             include=["documents", "metadatas", "distances"],
#         )
#         results = []
#         for cid, doc, meta, dist in zip(
#             raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
#         ):
#             results.append(
#                 SearchResult(
#                     chunk_id=cid,
#                     text=doc,
#                     score=1.0 - float(dist),  # cosine similarity
#                     method="vector",
#                     corpus=corpus,
#                     title=meta.get("title", ""),
#                     source_url=meta.get("source_url"),
#                     page_start=meta.get("page_start"),
#                     page_end=meta.get("page_end"),
#                     section=meta.get("section"),
#                     article_number=meta.get("article_number"),
#                     document_type=meta.get("document_type"),
#                     publication_year=meta.get("publication_year"),
#                 )
#             )
#         return results
