"""Nœud 7 — repli (déterministe) : répondre sans rien inventer, sans LLM.

Un message unique pour l'utilisateur (hors corpus, manque de couverture, document
restreint…) afin de ne jamais révéler l'existence d'un document non autorisé (F-SEC-05).
La cause n'est écrite que dans la trace, pour les administrateurs.
"""
from __future__ import annotations

from .. import config
from ..state import AgentState, event


def repli(state: AgentState) -> dict:
    status = "refus" if state.get("status") == "refus" else "je_ne_sais_pas"
    cause = state.get("repli_cause") or config.CAUSE_NO_VALID_SUB
    event(state, "repli", status=status, cause=cause)
    return {"final_answer": config.FALLBACK_MESSAGE, "citations": [], "status": status,
            "confidence": 0.0, "repli_cause": cause}
