"""Nœud 2 — orchestrateur (agent LLM) : route + question autonome + sous-questions.

Appelé une seule fois par requête. Il ne répond jamais de mémoire à une nouvelle
question juridique : `direct` uniquement pour une salutation, un remerciement, une
question sur le système, ou une réponse déjà présente (avec citations) dans l'historique.
"""
from __future__ import annotations

import re
from typing import Any

from .. import config
from ..llm import LLMClient, ask_json, register_heuristic
from ..state import AgentState, event, new_sub_result

SYSTEM = (
    "Tu es l'orchestrateur d'un assistant de recherche sur le droit et la réglementation "
    "marocains rédigés en français. Tu ne réponds JAMAIS de mémoire à une question juridique."
)
INSTRUCTIONS = """
1. Réécris la question en question autonome grâce à l'historique (standalone_question).
2. Choisis route :
   - "direct" : salutation, remerciement, question sur le système (réponds alors avec
     system_description, sans rien ajouter), ou réponse déjà présente dans l'historique
     AVEC ses citations (recopie-la dans direct_answer) ;
   - "recherche" : toute nouvelle question juridique ou réglementaire ;
   - "refus" : hors périmètre.
3. Si route = "recherche", décompose en 1 à 4 sous-questions si la demande en contient plusieurs.
Format : {"route": "...", "standalone_question": "...", "direct_answer": null,
          "sub_questions": [{"id": "q1", "text": "..."}]}
"""

SYSTEM_DESCRIPTION = (
    "Je suis un assistant de recherche sur les textes juridiques et réglementaires marocains "
    "rédigés en français. Je cherche les passages pertinents dans le corpus autorisé pour "
    "votre profil, je vérifie mes réponses et je cite mes sources. Je ne remplace pas un "
    "conseil juridique professionnel."
)


def _valid(out: dict[str, Any]) -> bool:
    return out.get("route") in config.ROUTES and isinstance(out.get("standalone_question", ""), str)


def normalize(out: dict[str, Any], question: str) -> dict[str, Any]:
    """Contrôle par code de la sortie du LLM (bornes, champs obligatoires)."""
    route = out["route"]
    standalone = (out.get("standalone_question") or question).strip()
    direct_answer = out.get("direct_answer") or None
    if route == "direct" and not (isinstance(direct_answer, str) and direct_answer.strip()):
        route = "recherche"  # pas de réponse directe exploitable : on cherche
    subs: list[dict[str, str]] = []
    if route == "recherche":
        for item in out.get("sub_questions") or []:
            text = item.get("text") if isinstance(item, dict) else item
            if isinstance(text, str) and text.strip():
                subs.append({"id": f"q{len(subs) + 1}", "text": text.strip()})
        subs = subs[: config.MAX_SUB_QUESTIONS] or [{"id": "q1", "text": standalone}]
    return {"route": route, "standalone_question": standalone,
            "direct_answer": direct_answer if route == "direct" else None, "sub_questions": subs}


def make_orchestrateur(llm: LLMClient):
    def orchestrateur(state: AgentState) -> dict:
        question = state["question"]
        data = {"question": question, "history": state.get("history", [])[-6:],
                "system_description": SYSTEM_DESCRIPTION}
        out = ask_json(state, llm, "orchestrateur", SYSTEM, INSTRUCTIONS, data,
                       retries=config.ORCHESTRATOR_JSON_RETRIES, validate=_valid)
        if out is None:
            decision = {"route": "recherche", "standalone_question": question,
                        "direct_answer": None, "sub_questions": [{"id": "q1", "text": question}]}
            event(state, "orchestrateur", fallback="json_invalide")
        else:
            decision = normalize(out, question)
        update: dict[str, Any] = dict(decision)
        update["sub_results"] = {s["id"]: new_sub_result(s["id"], s["text"])
                                 for s in decision["sub_questions"]}
        if decision["route"] == "refus":
            update.update(status="refus", repli_cause=config.CAUSE_OUT_OF_SCOPE)
        event(state, "orchestrateur", route=decision["route"],
              sub_questions=[s["text"] for s in decision["sub_questions"]])
        return update
    return orchestrateur


def route_after_orchestrateur(state: AgentState) -> str:
    return {"direct": "finalisation", "refus": "repli"}.get(state["route"], "chercheur")


# --------------------------------------------------------------------------- #
# Heuristique hors-ligne (backend "heuristic")
# --------------------------------------------------------------------------- #
_GREETING = re.compile(r"^\s*(bonjour|bonsoir|salut|coucou|hello|salam|merci|je vous remercie)\b", re.I)
_SYSTEM_Q = re.compile(r"(qui es[- ]tu|que (peux|sais)[- ]tu faire|comment (fonctionnes|marches)[- ]tu|"
                       r"qu'?est[- ]ce que (tu es|ce syst[eè]me))", re.I)
_REPEAT = re.compile(r"\b(rappelle|r[ée]p[eè]te|redis|reformule|r[ée]sume)[- ]?(moi)?\b", re.I)
_OUT_OF_SCOPE = re.compile(
    r"\b(recette|cuisine|football|match|m[ée]t[ée]o|film|s[ée]rie|chanson|po[eè]me|blague|"
    r"bitcoin|crypto|horoscope|jeu vid[ée]o|code python|programme en)\b", re.I)


@register_heuristic("orchestrateur")
def heuristic(data: dict[str, Any]) -> dict[str, Any]:
    question: str = data["question"].strip()
    history: list[dict[str, Any]] = data.get("history") or []
    if _GREETING.match(question) and len(question.split()) <= 6:
        answer = ("Avec plaisir." if question.lower().startswith(("merci", "je vous"))
                  else "Bonjour ! Posez-moi une question sur la réglementation marocaine.")
        return {"route": "direct", "standalone_question": question, "direct_answer": answer,
                "sub_questions": []}
    if _SYSTEM_Q.search(question):
        return {"route": "direct", "standalone_question": question,
                "direct_answer": SYSTEM_DESCRIPTION, "sub_questions": []}
    last = next((h for h in reversed(history) if h.get("role") == "assistant"), None)
    if _REPEAT.search(question) and last and last.get("citations"):
        return {"route": "direct", "standalone_question": question,
                "direct_answer": last["content"], "sub_questions": []}
    if _OUT_OF_SCOPE.search(question):
        return {"route": "refus", "standalone_question": question, "direct_answer": None,
                "sub_questions": []}
    standalone = question
    last_user = next((h for h in reversed(history) if h.get("role") == "user"), None)
    if last_user and len(question.split()) <= 6:
        standalone = f"{question} (à propos de : {last_user['content']})"
    parts = [p.strip(" ;") for p in re.split(r"\?|;", standalone) if len(p.strip(" ;").split()) >= 3]
    subs = [{"id": f"q{i}", "text": p + " ?"} for i, p in enumerate(parts[:4], start=1)]
    return {"route": "recherche", "standalone_question": standalone, "direct_answer": None,
            "sub_questions": subs or [{"id": "q1", "text": standalone}]}
