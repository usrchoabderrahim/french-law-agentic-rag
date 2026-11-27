"""Nœud 5 — generateur_final (agent LLM) : réponse finale avec citations [n].

La numérotation des passages [1], [2]… est faite par le code. Le LLM ne cite que ces
numéros, n'utilise aucune connaissance extérieure, écrit une section par sous-question
et « Ce que je n'ai pas trouvé : … » pour une sous-question insuffisante. En
régénération, il ne corrige que ce que controle_citations a signalé.
"""
from __future__ import annotations

from typing import Any

from .. import config
from ..llm import LLMClient, ask_json, register_heuristic
from ..state import AgentState, event
from ..text import fold, sentences

SYSTEM = ("Tu rédiges en français la réponse finale d'un assistant juridique marocain. "
          "Tu n'utilises que les réponses partielles validées et les passages numérotés fournis.")
INSTRUCTIONS = f"""
- Une section par sous-question validée ; cite les passages avec [n] (numéros reçus uniquement).
- Pour chaque sous-question insuffisante : « {config.NOT_FOUND_PREFIX}… ».
- citations : une entrée par [n] utilisé, avec chunk_id, document, page et un extrait
  recopié MOT POUR MOT depuis le passage.
- Si citation_feedback est non vide, corrige uniquement les erreurs signalées dans previous.
Format : {{"answer": "... [1] ...", "citations": [{{"n": 1, "chunk_id": "...",
          "document": "...", "page": 1, "extract": "..."}}]}}
"""


def number_passages(sub_results: dict) -> list[dict[str, Any]]:
    """Passages cités par les sous-questions validées, numérotés 1..n sans doublon."""
    numbered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sub in sub_results.values():
        if sub["status"] != "valide":
            continue
        for passage in sub["passages"]:
            if passage["chunk_id"] in sub["cited_chunk_ids"] and passage["chunk_id"] not in seen:
                seen.add(passage["chunk_id"])
                numbered.append({"n": len(numbered) + 1, **{k: passage[k] for k in
                                 ("chunk_id", "document", "page", "text")}})
    return numbered


def _valid(out: dict[str, Any]) -> bool:
    return isinstance(out.get("answer"), str) and isinstance(out.get("citations", []), list)


def make_generateur_final(llm: LLMClient):
    def generateur_final(state: AgentState) -> dict:
        subs = state["sub_results"]
        numbered = number_passages(subs)
        by_id = {p["chunk_id"]: p["n"] for p in numbered}
        insufficient = [s["text"] for s in subs.values() if s["status"] != "valide"]
        data = {
            "question": state.get("standalone_question") or state["question"],
            "validated": [{"sub_question": s["text"], "partial_answer": s["partial_answer"],
                           "passages": [by_id[c] for c in s["cited_chunk_ids"] if c in by_id]}
                          for s in subs.values() if s["status"] == "valide"],
            "passages": numbered,
            "insufficient": insufficient,
            "citation_feedback": state.get("citation_feedback") or [],
            "previous": ({"answer": state.get("final_answer"), "citations": state.get("citations")}
                         if state.get("citation_feedback") else None),
        }
        out = ask_json(state, llm, "generateur_final", SYSTEM, INSTRUCTIONS, data,
                       validate=_valid) or {"answer": "", "citations": []}
        answer = out["answer"].strip()
        for text in insufficient:                       # garantie par code de la règle
            if fold(text) not in fold(answer):
                answer += f"\n\n{config.NOT_FOUND_PREFIX}{text}"
        attempts = state.get("generation_attempts", 0) + 1
        event(state, "generateur_final", attempt=attempts, passages=len(numbered),
              citations=len(out["citations"]))
        return {"final_answer": answer, "citations": out["citations"],
                "generation_attempts": attempts}
    return generateur_final


# --------------------------------------------------------------------------- #
# Heuristique hors-ligne
# --------------------------------------------------------------------------- #
def _extract(partial: str, passage_text: str) -> str:
    """Première phrase de la réponse partielle présente telle quelle dans le passage."""
    for sent in sentences(partial):
        if sent in passage_text:
            return sent[:300]
    return sentences(passage_text)[0][:300] if sentences(passage_text) else passage_text[:300]


@register_heuristic("generateur_final")
def heuristic(data: dict[str, Any]) -> dict[str, Any]:
    passages = {p["n"]: p for p in data["passages"]}
    sections, citations = [], []
    for item in data["validated"]:
        marks = " ".join(f"[{n}]" for n in item["passages"])
        sections.append(f"**{item['sub_question']}**\n{item['partial_answer']} {marks}".strip())
        for n in item["passages"]:
            if all(c["n"] != n for c in citations):
                p = passages[n]
                citations.append({"n": n, "chunk_id": p["chunk_id"], "document": p["document"],
                                  "page": p["page"],
                                  "extract": _extract(item["partial_answer"], p["text"])})
    for text in data["insufficient"]:
        sections.append(f"{config.NOT_FOUND_PREFIX}{text}")
    return {"answer": "\n\n".join(sections), "citations": citations}
