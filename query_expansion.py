"""Expansion de requête (F-RET-05) : multi-query + HyDE, via un LLM local.

Principe :
- multi-query : le LLM reformule la question de N façons (synonymes, vocabulaire
  juridique) pour rattraper les chunks qui n'utilisent pas les mots de l'utilisateur ;
- HyDE : le LLM écrit une réponse hypothétique ; on cherche les chunks qui lui
  RESSEMBLENT (le vecteur d'une réponse est plus proche des passages que celui d'une question).

Si le LLM est indisponible, on dégrade proprement : liste vide / None, la recherche
continue avec la question d'origine uniquement.
"""
from __future__ import annotations

import json
import logging
import re

from .advanced_config import MULTI_QUERY_COUNT
from .config import CORPUS_LABELS
from .llm import LLM, LLMUnavailable

log = logging.getLogger(__name__)

SYSTEM = (
    "Tu es un assistant juridique spécialisé dans les textes marocains "
    "(Bulletin officiel, dahirs, décrets, arrêtés, règlements universitaires). "
    "Tu écris en français juridique clair."
)


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def parse_query_list(raw: str, n: int) -> list[str]:
    """Extrait une liste de questions d'une réponse LLM (JSON, sinon une ligne par question)."""
    items: list[str] = []
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [])
        if isinstance(data, list):
            items = [str(x) for x in data]
    except json.JSONDecodeError:
        for line in raw.splitlines():
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"')
            if line:
                items.append(line)
    return [i.strip() for i in items if i and i.strip()][: max(n, 0)]


class QueryExpander:
    def __init__(self, llm: LLM, n_queries: int = MULTI_QUERY_COUNT):
        self.llm = llm
        self.n_queries = n_queries

    def multi_query(self, query: str, corpus: str | None = None) -> list[str]:
        """Reformulations de `query` (sans la question d'origine, sans doublons)."""
        context = f"Corpus : {CORPUS_LABELS[corpus]}.\n" if corpus in CORPUS_LABELS else ""
        prompt = (
            f"{context}Question de l'utilisateur : {query}\n\n"
            f"Écris {self.n_queries} reformulations différentes de cette question pour une "
            "recherche documentaire : change le vocabulaire (synonymes, termes juridiques "
            "précis), garde exactement le même sens, ne réponds pas à la question.\n"
            'Réponds uniquement en JSON : {"queries": ["...", "..."]}'
        )
        try:
            raw = self.llm.generate(prompt, system=SYSTEM, json_mode=True)
        except LLMUnavailable as exc:
            log.warning("multi-query désactivé: %s", exc)
            return []
        seen = {_norm(query)}
        result: list[str] = []
        for candidate in parse_query_list(raw, self.n_queries * 2):
            key = _norm(candidate)
            if key and key not in seen:
                seen.add(key)
                result.append(candidate)
        return result[: self.n_queries]

    def hyde(self, query: str, corpus: str | None = None) -> str | None:
        """Passage hypothétique qui répondrait à la question (None si LLM indisponible)."""
        context = f"Corpus : {CORPUS_LABELS[corpus]}.\n" if corpus in CORPUS_LABELS else ""
        prompt = (
            f"{context}Question : {query}\n\n"
            "Rédige un court passage (3 à 5 phrases) tel qu'il pourrait apparaître dans un texte "
            "juridique ou réglementaire du corpus et qui répondrait à cette question. "
            "Utilise le vocabulaire et le ton d'un texte officiel. Reste générique sur les "
            "chiffres, dates et numéros de texte que tu ne connais pas. "
            "Pas d'introduction ni de commentaire : uniquement le passage."
        )
        try:
            text = self.llm.generate(prompt, system=SYSTEM, json_mode=False).strip()
        except LLMUnavailable as exc:
            log.warning("HyDE désactivé: %s", exc)
            return None
        return text or None
