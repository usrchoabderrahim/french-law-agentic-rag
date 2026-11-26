"""Nœud 1 — garde_entree (déterministe) : valide la requête avant tout appel au LLM."""
from __future__ import annotations

import re
import unicodedata

from .. import config
from ..state import AgentState, event

_INJECTION = [re.compile(p, re.I) for p in config.INJECTION_PATTERNS]


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text)


def check_input(question: str | None, user_role: str | None) -> str | None:
    """Renvoie la raison du refus, ou None si la requête est acceptée."""
    if not user_role:
        return "role_absent"                      # exigence T-14
    if user_role not in config.ROLES:
        return "role_inconnu"
    if not question or not question.strip():
        return "question_vide"
    if len(question) > config.MAX_QUESTION_CHARS:
        return "question_trop_longue"
    folded = _fold(question)
    if any(p.search(folded) for p in _INJECTION):
        return "injection_prompt"
    return None


def garde_entree(state: AgentState) -> dict:
    reason = check_input(state.get("question"), state.get("user_role"))
    if reason:
        event(state, "garde_entree", decision="refus", reason=reason)
        return {"status": "refus", "repli_cause": config.CAUSE_INPUT_REFUSED}
    event(state, "garde_entree", decision="ok")
    return {}


def route_after_garde(state: AgentState) -> str:
    return "repli" if state.get("status") == "refus" else "orchestrateur"
