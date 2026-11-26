"""Nœud 3 — chercheur (agent LLM + outil RAG) : une sous-question → passages + réponse partielle.

Tentative 1 : 1 à 3 requêtes (la sous-question et ses variantes).
Tentatives suivantes : lit feedback, missing_aspects, queries_tried, rejected_chunk_ids
et reformule. Une requête déjà essayée ne peut pas être répétée (contrôle par code).
"""
from __future__ import annotations

import re
from typing import Any

from .. import config
from ..llm import LLMClient, ask_json, register_heuristic
from ..state import AgentState, event
from ..text import fold, keywords, sentences
from ..tools import ToolError

SYSTEM = ("Tu es un chercheur juridique. Tu travailles uniquement à partir des passages "
          "fournis par l'outil search_corpus. Tu n'utilises aucune connaissance extérieure.")
QUERY_INSTRUCTIONS = """
Propose 1 à 3 requêtes de recherche pour la sous-question.
Tentative > 1 : tiens compte du feedback et des aspects manquants, reformule
(synonymes juridiques, HyDE, requête plus large ou plus étroite) et ne répète
AUCUNE requête de queries_tried.
Format : {"queries": ["...", "..."]}
"""
ANSWER_INSTRUCTIONS = """
Rédige une réponse partielle à la sous-question UNIQUEMENT à partir des passages.
Cite les chunk_id utilisés. Si aucun passage n'est utile, renvoie une réponse vide.
Format : {"partial_answer": "...", "cited_chunk_ids": ["..."]}
"""

_ARTICLE = re.compile(r"\barticles?\s+(premier|unique|\d+(?:\.\d+)*)", re.I)


def article_filter(text: str) -> dict[str, str] | None:
    """Filtre exact quand la sous-question cite un article précis."""
    match = _ARTICLE.search(text)
    if not match:
        return None
    value = match.group(1)
    return {"article_number": value.upper() if value.isalpha() else value}


def pick_sub(state: AgentState) -> str | None:
    subs = state["sub_results"]
    for wanted in ("en_cours", "en_attente"):
        for sub_id, sub in subs.items():
            if sub["status"] == wanted:
                return sub_id
    return None


def new_queries(proposed: list[Any], sub_text: str, tried: list[str], first: bool) -> list[str]:
    """Contrôle par code : sous-question en tête (tentative 1), aucune répétition, ≤ 3."""
    seen = {fold(q) for q in tried}
    out: list[str] = []
    for q in ([sub_text] if first else []) + [q for q in proposed if isinstance(q, str)]:
        key = fold(q)
        if q.strip() and key not in seen:
            seen.add(key)
            out.append(q.strip())
    return out[: config.MAX_QUERIES_PER_ATTEMPT]


def make_chercheur(llm: LLMClient, tool):
    def chercheur(state: AgentState) -> dict:
        sub_id = pick_sub(state)
        subs = dict(state["sub_results"])
        sub = dict(subs[sub_id])
        sub["attempts"] += 1
        verdict = sub.get("verdict") or {}
        first = sub["attempts"] == 1

        planned = ask_json(state, llm, "chercheur_requetes", SYSTEM, QUERY_INSTRUCTIONS, {
            "sub_question": sub["text"], "attempt": sub["attempts"],
            "queries_tried": sub["queries_tried"], "feedback": verdict.get("feedback", ""),
            "missing_aspects": verdict.get("missing_aspects", []),
            "rejected_chunk_ids": sub["rejected_chunk_ids"],
        }, validate=lambda o: isinstance(o.get("queries"), list))
        queries = new_queries((planned or {}).get("queries", []), sub["text"],
                              sub["queries_tried"], first)

        rejected = set(sub["rejected_chunk_ids"])
        found: dict[str, dict[str, Any]] = {}
        calls = 0
        filters = article_filter(sub["text"])
        for query in queries:
            for flt in ([filters, None] if filters else [None]):
                if calls >= config.MAX_TOOL_CALLS_PER_ATTEMPT:
                    break
                calls += 1
                try:
                    hits = tool.call("search_corpus", {"query": query, "top_k": config.TOP_K,
                                                       "filters": flt},
                                     user_role=state["user_role"], corpora=state["corpora"])
                except ToolError as exc:
                    event(state, "chercheur", sub=sub_id, erreur=str(exc))
                    hits = []
                for hit in hits:
                    if hit["chunk_id"] not in rejected:          # écarte les passages rejetés
                        found.setdefault(hit["chunk_id"], hit)
                if hits:
                    break  # le filtre exact a suffi : pas de requête non filtrée
        sub["queries_tried"] = sub["queries_tried"] + queries
        passages = list(found.values())[: 2 * config.TOP_K]

        answer, cited = "", []
        if passages:
            out = ask_json(state, llm, "chercheur_reponse", SYSTEM, ANSWER_INSTRUCTIONS, {
                "sub_question": sub["text"],
                "passages": [{k: p[k] for k in ("chunk_id", "document", "page", "text")}
                             for p in passages],
                "feedback": verdict.get("feedback", ""),
            }, validate=lambda o: isinstance(o.get("partial_answer", ""), str))
            if out:
                ids = {p["chunk_id"] for p in passages}
                answer = (out.get("partial_answer") or "").strip()
                cited = [c for c in out.get("cited_chunk_ids") or [] if c in ids]
        sub.update(passages=passages, partial_answer=answer, cited_chunk_ids=cited,
                   status="en_cours")
        subs[sub_id] = sub
        event(state, "chercheur", sub=sub_id, attempt=sub["attempts"], queries=queries,
              tool_calls=calls, passages=len(passages), cited=cited)
        return {"sub_results": subs, "current_sub": sub_id}
    return chercheur


# --------------------------------------------------------------------------- #
# Heuristiques hors-ligne
# --------------------------------------------------------------------------- #
SYNONYMS = {
    "inscription": ["admission", "immatriculation"], "inscrire": ["admission", "candidature"],
    "delai": ["durée", "date limite"], "duree": ["délai", "période"],
    "sanction": ["pénalité", "amende"], "amende": ["sanction", "pénalité"],
    "loi": ["dahir", "texte législatif"], "decret": ["arrêté", "texte réglementaire"],
    "etudiant": ["candidat", "inscrit"], "these": ["doctorat", "doctorant"],
    "examen": ["évaluation", "contrôle des connaissances"], "note": ["moyenne", "évaluation"],
    "module": ["unité d'enseignement", "élément de module"], "stage": ["projet de fin d'études"],
    "obligation": ["devoir", "engagement"], "droit": ["prérogative", "faculté"],
}


@register_heuristic("chercheur_requetes")
def heuristic_queries(data: dict[str, Any]) -> dict[str, Any]:
    text = data["sub_question"]
    words = [w for w in re.findall(r"\w+", text) if len(w) > 3]
    candidates = [" ".join(words)]
    for w in words:
        for syn in SYNONYMS.get(fold(w), []):
            candidates.append(text.replace(w, syn))
    if data.get("missing_aspects"):
        candidates.append(" ".join(words + list(data["missing_aspects"])))
    candidates.append("Le texte prévoit que " + " ".join(words))          # HyDE simplifié
    candidates.append(" ".join(sorted(words, key=len, reverse=True)[:2]))  # requête plus large
    tried = {fold(q) for q in data.get("queries_tried", [])}
    return {"queries": [q for q in candidates if q.strip() and fold(q) not in tried][:3]}


@register_heuristic("chercheur_reponse")
def heuristic_answer(data: dict[str, Any]) -> dict[str, Any]:
    """Réponse extractive : les phrases des passages qui partagent le plus de mots-clés."""
    target = keywords(data["sub_question"])
    scored = []
    for rank, passage in enumerate(data["passages"]):
        for pos, sent in enumerate(sentences(passage["text"])):
            score = len(target & keywords(sent))
            if score:
                scored.append((score, -rank, -pos, sent, passage["chunk_id"]))
    best = sorted(scored, reverse=True)[:3]
    if not best or best[0][0] < max(1, len(target) // 3):
        return {"partial_answer": "", "cited_chunk_ids": []}
    return {"partial_answer": " ".join(s[3] for s in best),
            "cited_chunk_ids": list(dict.fromkeys(s[4] for s in best))}
