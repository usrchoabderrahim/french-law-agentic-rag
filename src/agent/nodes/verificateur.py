"""Nœud 4 — verificateur (agent LLM + contrôle par code) : juge la réponse partielle.

Le LLM justifie, le code décide :
  * contrôle par code : chiffres, dates et numéros d'articles de la réponse doivent
    apparaître dans les passages cités ;
  * valid = supported_ratio ≥ 0,8 et relevance ≥ 0,6 ;
  * confidence = 0,5 × supported_ratio + 0,5 × relevance.
"""
from __future__ import annotations

import re
from typing import Any

from .. import config
from ..llm import LLMClient, ask_json, register_heuristic
from ..state import AgentState, event
from ..text import fold, keywords, sentences

SYSTEM = ("Tu es un vérificateur juridique strict. Tu contrôles qu'une réponse est étayée "
          "par les passages cités et qu'elle répond à la sous-question.")
INSTRUCTIONS = """
Évalue la réponse partielle :
- supported_ratio (0..1) : part des affirmations soutenues par le passage cité ;
- relevance (0..1) : la réponse répond-elle à la sous-question ?
- unsupported_claims, missing_aspects, rejected_chunk_ids (passages hors sujet), feedback.
Format : {"supported_ratio": 0.0, "relevance": 0.0, "unsupported_claims": [],
          "missing_aspects": [], "rejected_chunk_ids": [], "feedback": "..."}
"""

_MONTHS = r"janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[uû]t|septembre|octobre|novembre|d[ée]cembre"
_FACTS = re.compile(
    rf"\b\d{{1,2}}(?:er)?\s+(?:{_MONTHS})\s+\d{{4}}\b"          # dates en toutes lettres
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"                               # dates numériques
    r"|\barticles?\s+(?:premier|unique|\d+(?:[.\-]\d+)*)"          # numéros d'articles
    r"|\b\d+(?:[.,]\d+)*\b",                                        # chiffres
    re.I)


def extract_facts(text: str) -> list[str]:
    return list(dict.fromkeys(m.group(0) for m in _FACTS.finditer(text)))


def code_check(answer: str, cited_text: str) -> tuple[float, list[str]]:
    """Part des faits (chiffres, dates, articles) retrouvés dans les passages cités."""
    facts = extract_facts(answer)
    if not facts:
        return 1.0, []
    haystack = fold(cited_text)
    missing = [f for f in facts if fold(f) not in haystack]
    return (len(facts) - len(missing)) / len(facts), missing


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _valid_json(out: dict[str, Any]) -> bool:
    return "supported_ratio" in out and "relevance" in out


def decide(supported_ratio: float, relevance: float) -> tuple[bool, float]:
    valid = supported_ratio >= config.SUPPORTED_RATIO_MIN and relevance >= config.RELEVANCE_MIN
    return valid, round(0.5 * supported_ratio + 0.5 * relevance, 4)


def make_verificateur(llm: LLMClient):
    def verificateur(state: AgentState) -> dict:
        sub_id = state["current_sub"]
        subs = dict(state["sub_results"])
        sub = dict(subs[sub_id])
        answer = sub["partial_answer"]
        cited = [p for p in sub["passages"] if p["chunk_id"] in sub["cited_chunk_ids"]]

        if not answer or not cited:
            verdict = {"supported_ratio": 0.0, "relevance": 0.0, "unsupported_claims": [],
                       "missing_aspects": [sub["text"]], "rejected_chunk_ids":
                       [p["chunk_id"] for p in sub["passages"]],
                       "feedback": "Aucune réponse étayée : reformuler la recherche."}
        else:
            out = ask_json(state, llm, "verificateur", SYSTEM, INSTRUCTIONS, {
                "sub_question": sub["text"], "partial_answer": answer,
                "cited_passages": [{k: p[k] for k in ("chunk_id", "text")} for p in cited],
                "other_passages": [p["chunk_id"] for p in sub["passages"] if p not in cited],
            }, validate=_valid_json) or {"supported_ratio": 0, "relevance": 0,
                                         "feedback": "verdict LLM invalide"}
            ratio, missing_facts = code_check(answer, " ".join(p["text"] for p in cited))
            ids = {p["chunk_id"] for p in sub["passages"]}
            verdict = {
                "supported_ratio": min(_clamp(out.get("supported_ratio")), ratio),
                "relevance": _clamp(out.get("relevance")),
                "unsupported_claims": list(out.get("unsupported_claims") or [])
                + [f"élément absent du passage cité : {f}" for f in missing_facts],
                "missing_aspects": list(out.get("missing_aspects") or []),
                "rejected_chunk_ids": [c for c in out.get("rejected_chunk_ids") or [] if c in ids],
                "feedback": str(out.get("feedback") or ""),
            }
        verdict["valid"], verdict["confidence"] = decide(verdict["supported_ratio"],
                                                         verdict["relevance"])
        if verdict["valid"]:
            status = "valide"
        elif sub["attempts"] < config.MAX_RESEARCH_ATTEMPTS:
            status = "en_cours"          # nouvelle tentative possible
        else:
            status = "insuffisant"
        sub.update(verdict=verdict, status=status, rejected_chunk_ids=list(dict.fromkeys(
            sub["rejected_chunk_ids"] + verdict["rejected_chunk_ids"])))
        subs[sub_id] = sub
        event(state, "verificateur", sub=sub_id, attempt=sub["attempts"], status=status,
              supported_ratio=verdict["supported_ratio"], relevance=verdict["relevance"],
              confidence=verdict["confidence"])
        return {"sub_results": subs}
    return verificateur


# --------------------------------------------------------------------------- #
# Heuristique hors-ligne
# --------------------------------------------------------------------------- #
@register_heuristic("verificateur")
def heuristic(data: dict[str, Any]) -> dict[str, Any]:
    passages = data["cited_passages"]
    corpus = fold(" ".join(p["text"] for p in passages))
    claims = sentences(data["partial_answer"]) or [data["partial_answer"]]
    unsupported = [c for c in claims if fold(c) not in corpus
                   and len(keywords(c) - keywords(corpus)) > len(keywords(c)) * 0.2]
    target = keywords(data["sub_question"])
    covered = target & keywords(data["partial_answer"])
    relevance = len(covered) / len(target) if target else 0.0
    missing = sorted(target - covered)
    rejected = [p["chunk_id"] for p in passages if not target & keywords(p["text"])]
    return {
        "supported_ratio": round(1 - len(unsupported) / len(claims), 4),
        "relevance": round(relevance, 4),
        "unsupported_claims": unsupported, "missing_aspects": missing,
        "rejected_chunk_ids": rejected,
        "feedback": ("Réponse étayée." if not unsupported and relevance >= config.RELEVANCE_MIN
                     else f"Aspects manquants : {', '.join(missing) or 'aucun'} ; "
                          f"{len(unsupported)} affirmation(s) non étayée(s)."),
    }
