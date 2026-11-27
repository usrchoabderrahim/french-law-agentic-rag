"""Nœud 6 — controle_citations (déterministe) : filet de sécurité, sans LLM.

Contrôles :
  1. chaque [n] du texte existe dans la table citations ;
  2. chaque chunk_id cité appartient aux passages validés (rien d'inventé) ;
  3. chaque citation a document, page et extrait ;
  4. l'extrait cité figure bien dans le texte du passage.
"""
from __future__ import annotations

import re

from ..state import AgentState, event
from ..text import fold

_MARK = re.compile(r"\[(\d+)\]")


def validated_passages(sub_results: dict) -> dict[str, str]:
    return {p["chunk_id"]: p["text"] for s in sub_results.values() if s["status"] == "valide"
            for p in s["passages"]}


def check_citations(answer: str, citations: list, sub_results: dict) -> list[str]:
    errors: list[str] = []
    if not answer.strip():
        return ["réponse vide"]
    table = {}
    for c in citations:
        if not isinstance(c, dict) or not isinstance(c.get("n"), int):
            errors.append(f"citation mal formée : {c!r}"[:200])
        else:
            table[c["n"]] = c
    for n in sorted({int(m) for m in _MARK.findall(answer)}):
        if n not in table:
            errors.append(f"[{n}] utilisé dans le texte mais absent de la table des citations")
    allowed = validated_passages(sub_results)
    for n, c in sorted(table.items()):
        chunk_id = c.get("chunk_id")
        if chunk_id not in allowed:
            errors.append(f"[{n}] chunk_id {chunk_id!r} absent des passages validés (inventé)")
            continue
        for field in ("document", "page", "extract"):
            if c.get(field) in (None, ""):
                errors.append(f"[{n}] champ « {field} » manquant")
        extract = c.get("extract")
        if isinstance(extract, str) and extract and fold(extract) not in fold(allowed[chunk_id]):
            errors.append(f"[{n}] l'extrait cité ne figure pas dans le passage {chunk_id}")
    return errors


def make_controle_citations(llm=None):  # signature commune ; aucun LLM utilisé
    def controle_citations(state: AgentState) -> dict:
        errors = check_citations(state.get("final_answer", ""), state.get("citations") or [],
                                 state["sub_results"])
        event(state, "controle_citations", ok=not errors, errors=errors,
              attempt=state.get("generation_attempts", 0))
        return {"citation_feedback": errors}
    return controle_citations
