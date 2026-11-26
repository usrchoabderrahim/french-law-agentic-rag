"""Outil RAG `search_corpus` utilisé par le chercheur (contribution « recherche »).

Sécurité :
  * liste blanche d'outils : seul `search_corpus` est appelable ;
  * le `user_role` est ajouté par ce wrapper depuis l'état : le LLM ne le voit pas et
    ne peut pas le modifier (tout argument `user_role`/`role` fourni est ignoré) ;
  * le filtrage par rôle est fait AVANT le score par la couche retrieval (F-SEC-04) ;
  * timeout par appel.

Recherche : réutilise `src.retrieval.search.Retriever` (vector | bm25 | hybrid RRF) sur
les deux corpus (`fstbm`, `ocr`), puis fusionne les corpus par RRF (`rrf_fuse`).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Callable

import numpy as np

from src.retrieval.hybrid import rrf_fuse
from src.retrieval.loader import IndexedChunk, load_all
from src.retrieval.models import SearchResult
from src.retrieval.search import Retriever
from src.retrieval.vector_store import build_where

from . import config

ALLOWED_TOOLS = frozenset({"search_corpus"})
ALLOWED_ARGS = frozenset({"query", "top_k", "filters"})


class ToolError(RuntimeError):
    pass


def to_result(chunk: IndexedChunk, score: float, method: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk.chunk_id, text=chunk.text, score=score, method=method, corpus=chunk.corpus,
        title=chunk.title, source_url=chunk.source_url, page_start=chunk.page_start,
        page_end=chunk.page_end, section=chunk.section, article_number=chunk.article_number,
        document_type=chunk.document_type, publication_year=chunk.publication_year,
    )


class NumpyVectorStore:
    """Recherche vectorielle exacte en mémoire, même API que `ChromaStore.search`.

    Les corpus sont petits (44 + 981 vecteurs bge-m3 normalisés) : un produit scalaire
    suffit et évite d'exiger un serveur ou un index Chroma pour lancer l'agent.
    """

    def __init__(self, corpora: dict[str, list[IndexedChunk]] | None = None):
        self.corpora = corpora or load_all()
        self.matrices = {name: np.stack([c.embedding for c in chunks])
                         for name, chunks in self.corpora.items()}

    def search(self, query_vector, corpus: str, role: str | None, k: int,
               filters: dict[str, Any] | None = None) -> list[SearchResult]:
        build_where(role, filters)  # même contrôle de rôle que Chroma (PermissionError)
        if corpus not in self.corpora:
            raise ValueError(f"Unknown corpus '{corpus}'")
        chunks = self.corpora[corpus]
        sims = self.matrices[corpus] @ np.asarray(query_vector, dtype=np.float32)
        out: list[SearchResult] = []
        for idx in np.argsort(-sims):
            chunk = chunks[idx]
            if role not in chunk.allowed_roles:
                continue
            if filters and not all(getattr(chunk, f, None) == v for f, v in filters.items()):
                continue
            out.append(to_result(chunk, float(sims[idx]), "vector"))
            if len(out) == k:
                break
        return out


def to_passage(result: SearchResult) -> dict[str, Any]:
    return {"chunk_id": result.chunk_id, "text": result.text, "document": result.title,
            "page": result.page_start, "corpus": result.corpus, "score": round(result.score, 6),
            "article_number": result.article_number, "source_url": result.source_url}


class SearchTool:
    """Exécute les appels d'outil demandés par le chercheur."""

    def __init__(self, retriever: Retriever, mode: str = "bm25",
                 timeout: float = config.TOOL_TIMEOUT_SECONDS):
        self.retriever = retriever
        self.mode = mode
        self.timeout = timeout
        self._pool = ThreadPoolExecutor(max_workers=2)

    def call(self, name: str, args: dict[str, Any], *, user_role: str,
             corpora: list[str]) -> list[dict[str, Any]]:
        if name not in ALLOWED_TOOLS:
            raise ToolError(f"outil non autorisé : {name}")
        clean = {k: v for k, v in args.items() if k in ALLOWED_ARGS}  # user_role ignoré
        future = self._pool.submit(self.search_corpus, user_role=user_role, corpora=corpora, **clean)
        try:
            return future.result(timeout=self.timeout)
        except FutureTimeout as exc:
            raise ToolError("timeout de l'outil search_corpus") from exc

    def search_corpus(self, query: str, user_role: str, corpora: list[str],
                      top_k: int = config.TOP_K, filters: dict[str, Any] | None = None
                      ) -> list[dict[str, Any]]:
        rankings = [self.retriever.search(query, corpus, user_role, k=top_k, mode=self.mode,
                                          filters=filters or None)
                    for corpus in corpora if corpus in self.retriever.corpora]
        if not rankings:
            return []
        fused = rankings[0][:top_k] if len(rankings) == 1 else rrf_fuse(rankings, top_k)
        return [to_passage(r) for r in fused]


def _encoder() -> Callable[[str], list[float]] | None:
    if os.getenv("RAG_SEARCH_MODE", "auto") == "bm25":
        return None
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return None
    from src.retrieval.query_encoder import QueryEncoder
    encoder = QueryEncoder()
    encoder.encode("préchauffage")  # charge bge-m3 ici, jamais pendant un appel d'outil minuté
    return encoder.encode


def build_search_tool() -> SearchTool:
    """Hybride (bge-m3 + BM25, RRF) si sentence-transformers est installé, sinon BM25."""
    encode = _encoder()
    corpora = load_all()
    store: Any = NumpyVectorStore(corpora)
    if os.getenv("RAG_VECTOR_STORE") == "chroma":
        from src.retrieval.vector_store import ChromaStore
        store = ChromaStore()
        store.build_all()
    retriever = Retriever(vector_store=store, encode=encode or (lambda q: []))
    return SearchTool(retriever, mode="hybrid" if encode else "bm25")