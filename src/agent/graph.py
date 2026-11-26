"""Moteur du graphe agentique : nœuds, arêtes conditionnelles et boucle d'exécution.

    __start__ → garde_entree ─ok→ orchestrateur ─recherche→ [chercheur ⇄ verificateur] ×k
                     │refus          │direct │refus          │≥1 valide        │aucune valide
                     ▼               ▼       ▼               ▼                 ▼
                   repli        finalisation repli     generateur_final ⇄ controle_citations
                     │                                        (invalide, < G)   │ok   │échec
                     └──────────────→ finalisation ←────────────────────────────┘     ▼
                                          │                                         repli
                                       __end__

Les nœuds de recherche (chercheur, verificateur) et de génération (generateur_final,
controle_citations) sont chargés à la demande : s'ils sont absents de l'installation,
la requête part en repli (cause « noeud_indisponible » dans la trace).
"""
from __future__ import annotations

import importlib
from typing import Any, Callable

from . import config
from .llm import BudgetExceeded, LLMClient, LLMUnavailable, get_llm
from .nodes.finalisation import finalisation
from .nodes.garde_entree import garde_entree, route_after_garde
from .nodes.orchestrateur import make_orchestrateur, route_after_orchestrateur
from .nodes.repli import repli
from .state import AgentState, event, new_state

END = "__end__"
MAX_STEPS = 200


class NodeUnavailable(RuntimeError):
    pass


def route_after_verificateur(state: AgentState) -> str:
    subs = state.get("sub_results", {}).values()
    if any(s["status"] in ("en_attente", "en_cours") for s in subs):
        return "chercheur"            # non valide et < N, ou sous-question suivante
    if any(s["status"] == "valide" for s in subs):
        return "generateur_final"     # ≥ 1 valide
    return "repli"                    # aucune valide


def route_after_controle(state: AgentState) -> str:
    if not state.get("citation_feedback"):
        return "finalisation"
    if state.get("generation_attempts", 0) < config.MAX_GENERATION_ATTEMPTS:
        return "generateur_final"     # invalide, < G
    return "repli"                    # échec


ROUTERS: dict[str, Callable[[AgentState], str]] = {
    "garde_entree": route_after_garde,
    "orchestrateur": route_after_orchestrateur,
    "chercheur": lambda s: "verificateur",
    "verificateur": route_after_verificateur,
    "generateur_final": lambda s: "controle_citations",
    "controle_citations": route_after_controle,
    "repli": lambda s: "finalisation",
    "finalisation": lambda s: END,
}

# Cause écrite dans la trace quand une arête mène au repli.
REPLI_CAUSES = {
    "verificateur": config.CAUSE_NO_VALID_SUB,
    "controle_citations": config.CAUSE_CITATIONS,
}

# Nœuds fournis par les autres contributions : (module, fabrique)
LAZY_NODES = {
    "chercheur": ("src.agent.nodes.chercheur", "make_chercheur"),
    "verificateur": ("src.agent.nodes.verificateur", "make_verificateur"),
    "generateur_final": ("src.agent.nodes.generateur_final", "make_generateur_final"),
    "controle_citations": ("src.agent.nodes.controle_citations", "make_controle_citations"),
}


class AgenticRAG:
    """Point d'entrée : `AgenticRAG().run(question, user_role)`."""

    def __init__(self, llm: LLMClient | None = None, search_tool: Any | None = None):
        self.llm = llm or get_llm()
        self._search_tool = search_tool
        self.nodes: dict[str, Callable[[AgentState], dict]] = {
            "garde_entree": garde_entree,
            "orchestrateur": make_orchestrateur(self.llm),
            "repli": repli,
            "finalisation": finalisation,
        }

    @property
    def search_tool(self):
        if self._search_tool is None:
            try:
                from .tools import build_search_tool
            except ImportError as exc:
                raise NodeUnavailable("outil search_corpus absent") from exc
            self._search_tool = build_search_tool()
        return self._search_tool

    def _node(self, name: str) -> Callable[[AgentState], dict]:
        if name not in self.nodes:
            module_name, factory = LAZY_NODES[name]
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as exc:
                raise NodeUnavailable(name) from exc
            make = getattr(module, factory)
            self.nodes[name] = make(self.llm, self.search_tool) if name == "chercheur" else make(self.llm)
        return self.nodes[name]

    def run(self, question: str, user_role: str | None, history: list[dict] | None = None,
            corpora: list[str] | None = None) -> AgentState:
        state = new_state(question, user_role, history, corpora)
        node = "garde_entree"
        for _ in range(MAX_STEPS):
            if node == END:
                return state
            try:
                state.update(self._node(node)(state))
                nxt = ROUTERS[node](state)
                if nxt == "repli" and not state.get("repli_cause"):
                    state["repli_cause"] = REPLI_CAUSES.get(node)
            except BudgetExceeded as exc:
                event(state, node, erreur="budget", detail=str(exc))
                state["repli_cause"], nxt = config.CAUSE_BUDGET, "repli"
            except LLMUnavailable as exc:
                event(state, node, erreur="llm_indisponible", detail=str(exc)[:200])
                state["repli_cause"], nxt = config.CAUSE_LLM, "repli"
            except NodeUnavailable as exc:
                event(state, node, erreur="noeud_indisponible", detail=str(exc))
                state["repli_cause"], nxt = config.CAUSE_MISSING_NODE, "repli"
            node = nxt
        raise RuntimeError("Le graphe n'a pas atteint __end__ (boucle ?)")


def answer(question: str, user_role: str | None, **kwargs: Any) -> dict[str, Any]:
    """Raccourci : renvoie la sortie utilisateur (réponse, statut, confiance, citations)."""
    state = AgenticRAG(**{k: kwargs.pop(k) for k in ("llm", "search_tool") if k in kwargs}).run(
        question, user_role, **kwargs)
    return {key: state.get(key) for key in
            ("final_answer", "status", "confidence", "citations", "route", "trace")}
