"""Nœud 8 — finalisation (déterministe) : statut final, confiance globale, avertissement."""
from __future__ import annotations

import time

from .. import config
from ..state import AgentState, event


def global_confidence(sub_results: dict) -> float:
    """Moyenne des confidence validées × part de sous-questions validées."""
    subs = list(sub_results.values())
    valid = [s for s in subs if s.get("status") == "valide"]
    if not subs or not valid:
        return 0.0
    mean = sum(s["verdict"]["confidence"] for s in valid) / len(valid)
    return round(mean * len(valid) / len(subs), 4)


def finalisation(state: AgentState) -> dict:
    update: dict = {}
    if state.get("status") in ("refus", "je_ne_sais_pas"):          # conserve le repli
        update["confidence"] = 0.0
        answer = state["final_answer"]
    elif state.get("route") == "direct":
        update.update(status="reponse", confidence=None)              # None pour une réponse directe
        answer = state.get("direct_answer") or ""
    else:
        subs = state.get("sub_results", {}).values()
        insufficient = any(s.get("status") != "valide" for s in subs)
        update.update(status="partielle" if insufficient else "reponse",
                      confidence=global_confidence(state["sub_results"]))
        answer = state.get("final_answer", "")
    if update.get("status") in ("reponse", "partielle"):
        answer = f"{answer}\n\n⚠️ {config.WARNING_MESSAGE}"
    update["final_answer"] = answer

    trace = state["trace"]
    trace["latency_ms"] = round((time.time() - trace["started_at"]) * 1000, 1)
    trace["closed"] = True
    event(state, "finalisation", status=update.get("status", state.get("status")),
          confidence=update["confidence"], llm_calls=trace["llm_calls"], tokens=trace["tokens"])
    return update
