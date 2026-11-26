"""État partagé du graphe (un dict) et petits utilitaires de trace.

Chaque nœud lit l'état et renvoie un dict de mises à jour ; le moteur (graph.py)
fusionne ces mises à jour. Les champs suivent les « Lit / Écrit » de la spécification.
"""
from __future__ import annotations

import time
from typing import Any, TypedDict


class Verdict(TypedDict, total=False):
    supported_ratio: float
    relevance: float
    confidence: float
    valid: bool
    unsupported_claims: list[str]
    missing_aspects: list[str]
    rejected_chunk_ids: list[str]
    feedback: str


class SubResult(TypedDict, total=False):
    id: str
    text: str
    status: str                 # en_attente | en_cours | valide | insuffisant
    attempts: int
    queries_tried: list[str]
    passages: list[dict[str, Any]]
    partial_answer: str
    cited_chunk_ids: list[str]
    rejected_chunk_ids: list[str]   # cumul des rejets du vérificateur
    verdict: Verdict | None


class AgentState(TypedDict, total=False):
    # entrée
    question: str
    user_role: str | None
    history: list[dict[str, Any]]
    corpora: list[str]
    # orchestrateur
    standalone_question: str
    route: str
    direct_answer: str | None
    sub_questions: list[dict[str, str]]
    # recherche
    sub_results: dict[str, SubResult]
    current_sub: str | None     # sous-question traitée par chercheur → verificateur
    # génération
    final_answer: str
    citations: list[dict[str, Any]]
    citation_feedback: list[str]
    generation_attempts: int
    # sortie
    status: str                 # reponse | partielle | je_ne_sais_pas | refus
    confidence: float | None
    repli_cause: str | None
    trace: dict[str, Any]


def new_state(
    question: str,
    user_role: str | None,
    history: list[dict[str, Any]] | None = None,
    corpora: list[str] | None = None,
) -> AgentState:
    return AgentState(
        question=question,
        user_role=user_role,
        history=list(history or []),
        corpora=list(corpora or ["fstbm", "ocr"]),
        sub_results={},
        citations=[],
        citation_feedback=[],
        generation_attempts=0,
        status="en_cours",
        confidence=None,
        repli_cause=None,
        trace={"events": [], "started_at": time.time(), "llm_calls": 0, "tokens": 0},
    )


def event(state: AgentState, node: str, **details: Any) -> dict[str, Any]:
    """Ajoute un évènement à la trace (mutation en place, renvoyée pour la fusion)."""
    trace = state["trace"]
    trace["events"].append({"node": node, "t": round(time.time() - trace["started_at"], 4), **details})
    return trace


def new_sub_result(sub_id: str, text: str) -> SubResult:
    return SubResult(
        id=sub_id, text=text, status="en_attente", attempts=0, queries_tried=[],
        passages=[], partial_answer="", cited_chunk_ids=[], rejected_chunk_ids=[], verdict=None,
    )
