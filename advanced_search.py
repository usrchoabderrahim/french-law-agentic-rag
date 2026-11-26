"""Recherche avancée (sous-tâche B) : expansion de requête + HyDE + GraphRAG, fusionnés par RRF.

Chaîne complète (voir AdvancedSearchPipeline) :

    question
      ├─ [multi-query]  N reformulations (LLM)      ┐
      ├─ question d'origine                          ├─ chaque variante -> recherche de base
      ├─ [HyDE]  réponse hypothétique -> vecteur     │   (hybride BM25+vecteur, rôle + filtres appliqués)
      └─ [GraphRAG] entités -> voisins -> chunks     ┘
                         │
                 RRF pondéré  ->  re-ranking bge-reranker (avec la question ORIGINALE)  ->  top-k

Tout passe par `Retriever.search` / `ChromaStore.search` / `GraphStore.search` : le filtre par
rôle (F-SEC-04) est donc appliqué AVANT toute recherche, pour chaque variante.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any

from .advanced_config import (
    WEIGHT_GRAPH,
    WEIGHT_HYDE,
    WEIGHT_MULTI_QUERY,
    WEIGHT_ORIGINAL,
)
from .config import CANDIDATES_PER_METHOD, DEFAULT_TOP_K, RRF_K
from .graph_rag import GraphStore
from .llm import LLM
from .models import SearchResult
from .query_expansion import QueryExpander
from .search import Retriever
from .vector_store import check_role

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdvancedOptions:
    multi_query: bool = True
    hyde: bool = True
    graph: bool = True
    rerank: bool = True
    base_mode: str = "hybrid"  # méthode utilisée pour chaque variante de la question

    def label(self) -> str:
        parts = [self.base_mode]
        parts += [n for n, on in (("multi", self.multi_query), ("hyde", self.hyde), ("graph", self.graph)) if on]
        if self.rerank:
            parts.append("rerank")
        return "+".join(parts)


@dataclass
class SearchTrace:
    """Ce qui s'est passé pendant la recherche (pour la démo, le debug et le rapport)."""

    query: str
    reformulations: list[str] = field(default_factory=list)
    hyde_text: str | None = None
    graph_seeds: int = 0
    ranking_sizes: dict[str, int] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)


def weighted_rrf(
    rankings: list[tuple[float, list[SearchResult]]], k: int, rrf_k: int = RRF_K
) -> list[SearchResult]:
    """score(chunk) = somme sur les classements de  poids / (rrf_k + rang)."""
    scores: dict[str, float] = {}
    best: dict[str, SearchResult] = {}
    for weight, ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + weight / (rrf_k + rank)
            best.setdefault(result.chunk_id, result)
    ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))[:k]
    return [replace(best[cid], score=scores[cid], method="advanced") for cid in ordered]


class AdvancedRetriever:
    """Sous-tâche B sans re-ranking : variantes de requête + HyDE + graphe, fusionnés par RRF."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLM | None = None,
        graph_store: GraphStore | None = None,
        expander: QueryExpander | None = None,
    ):
        self.retriever = retriever
        self.expander = expander or (QueryExpander(llm) if llm is not None else None)
        self.graph_store = graph_store or GraphStore(retriever.corpora)

    def search_with_trace(
        self,
        query: str,
        corpus: str,
        role: str | None,
        k: int = DEFAULT_TOP_K,
        options: AdvancedOptions = AdvancedOptions(),
        document_type: str | None = None,
        filters: dict[str, Any] | None = None,
        pool: int = CANDIDATES_PER_METHOD,
    ) -> tuple[list[SearchResult], SearchTrace]:
        if not query.strip():
            raise ValueError("Empty query")
        check_role(role)  # refus AVANT tout appel au LLM
        all_filters = dict(filters or {})
        if document_type:
            all_filters["document_type"] = document_type
        all_filters = all_filters or None

        trace = SearchTrace(query=query)
        rankings: list[tuple[float, list[SearchResult]]] = []

        def timed(name: str, fn):
            start = time.perf_counter()
            value = fn()
            trace.seconds[name] = round(time.perf_counter() - start, 3)
            return value

        # 1) question d'origine : la base reste toujours présente
        base = timed("base", lambda: self.retriever.search(
            query, corpus, role, k=pool, mode=options.base_mode, filters=all_filters))
        rankings.append((WEIGHT_ORIGINAL, base))
        trace.ranking_sizes["original"] = len(base)

        # 2) multi-query
        if options.multi_query and self.expander is not None:
            trace.reformulations = timed("multi_query_llm", lambda: self.expander.multi_query(query, corpus))
            for i, variant in enumerate(trace.reformulations, start=1):
                found = self.retriever.search(
                    variant, corpus, role, k=pool, mode=options.base_mode, filters=all_filters)
                rankings.append((WEIGHT_MULTI_QUERY, found))
                trace.ranking_sizes[f"multi_{i}"] = len(found)

        # 3) HyDE : on encode la réponse hypothétique (même modèle bge-m3) et on cherche en vectoriel
        if options.hyde and self.expander is not None:
            trace.hyde_text = timed("hyde_llm", lambda: self.expander.hyde(query, corpus))
            if trace.hyde_text:
                found = self.retriever.vector_store.search(
                    self.retriever.encode(trace.hyde_text), corpus, role, pool, all_filters)
                rankings.append((WEIGHT_HYDE, found))
                trace.ranking_sizes["hyde"] = len(found)

        # 4) GraphRAG
        if options.graph:
            graph_hits = timed("graph", lambda: self.graph_store.search(query, corpus, role, pool, all_filters))
            trace.graph_seeds = len(self.graph_store.graph(corpus, role).find_seeds(query))
            if graph_hits:
                rankings.append((WEIGHT_GRAPH, graph_hits))
            trace.ranking_sizes["graph"] = len(graph_hits)

        fused = weighted_rrf(rankings, max(pool, k))
        return fused, trace

    def search(self, query: str, corpus: str, role: str | None, k: int = DEFAULT_TOP_K,
               options: AdvancedOptions = AdvancedOptions(), **kwargs: Any) -> list[SearchResult]:
        return self.search_with_trace(query, corpus, role, k, options, **kwargs)[0][:k]


class AdvancedSearchPipeline:
    """Point d'entrée de la sous-tâche B : AdvancedRetriever + re-ranking final."""

    def __init__(self, advanced: AdvancedRetriever, reranker=None):
        self.advanced = advanced
        self._reranker = reranker

    @classmethod
    def from_pipeline(cls, pipeline, llm: LLM | None = None) -> "AdvancedSearchPipeline":
        """Réutilise le LegalSearchPipeline déjà construit (sous-tâche A) : index et reranker partagés."""
        if llm is None:
            from .llm import OllamaClient

            llm = OllamaClient()
        return cls(AdvancedRetriever(pipeline.retriever, llm=llm), reranker=pipeline.reranker)

    @property
    def reranker(self):
        if self._reranker is None:
            from .reranker import Reranker  # import tardif : charge bge-reranker seulement si utile

            self._reranker = Reranker()
        return self._reranker

    def search_with_trace(
        self,
        query: str,
        corpus: str,
        role: str | None,
        k: int = DEFAULT_TOP_K,
        options: AdvancedOptions = AdvancedOptions(),
        document_type: str | None = None,
        filters: dict[str, Any] | None = None,
        candidates: int = CANDIDATES_PER_METHOD,
    ) -> tuple[list[SearchResult], SearchTrace]:
        pool = max(candidates, k)
        fused, trace = self.advanced.search_with_trace(
            query, corpus, role, k, options, document_type, filters, pool)
        if options.rerank and fused:
            start = time.perf_counter()
            # re-ranking avec la question ORIGINALE : les reformulations ne servent qu'à trouver des candidats
            fused = self.reranker.rerank(query, fused, k=k)
            trace.seconds["rerank"] = round(time.perf_counter() - start, 3)
        return fused[:k], trace

    def search(self, query: str, corpus: str, role: str | None, k: int = DEFAULT_TOP_K,
               options: AdvancedOptions = AdvancedOptions(), **kwargs: Any) -> list[SearchResult]:
        return self.search_with_trace(query, corpus, role, k, options, **kwargs)[0]
