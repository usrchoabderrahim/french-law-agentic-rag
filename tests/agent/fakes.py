"""Doublures de test partagées : LLM scripté, outil de recherche factice, garde données."""
from __future__ import annotations

import json
from typing import Any, Callable

import pytest

from src.agent.llm import _DATA_RE, _TASK_RE, HeuristicLLM
from src.retrieval.config import CORPORA

requires_data = pytest.mark.skipif(
    not all((p / "embeddings.npy").exists() for p in CORPORA.values()),
    reason="jeux d'embeddings absents de data/embeddings (voir docs/AGENTIC_RAG.md)",
)

Reply = str | dict | Callable[[dict], Any]


class ScriptedLLM:
    """Réponses imposées par tâche (consommées dans l'ordre), sinon moteur heuristique."""

    def __init__(self, script: dict[str, list[Reply]] | None = None):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.calls: list[tuple[str, dict]] = []
        self.fallback = HeuristicLLM()

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        task = _TASK_RE.search(prompt).group(1)
        data = json.loads(_DATA_RE.search(prompt).group(1))
        self.calls.append((task, data))
        queue = self.script.get(task)
        if not queue:
            return self.fallback.generate(prompt, system, json_mode)
        reply = queue.pop(0) if len(queue) > 1 else queue[0]  # la dernière réponse se répète
        if callable(reply):
            reply = reply(data)
        return reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)

    def tasks(self) -> list[str]:
        return [t for t, _ in self.calls]


def passage(chunk_id: str, text: str, page: int = 1, document: str = "Doc test") -> dict:
    return {"chunk_id": chunk_id, "text": text, "document": document, "page": page,
            "corpus": "fstbm", "score": 1.0, "article_number": None, "source_url": None}


class FakeSearchTool:
    """Même interface que SearchTool.call ; renvoie des passages fixes, journalise les appels."""

    def __init__(self, passages: list[dict] | Callable[[str], list[dict]]):
        self.passages = passages
        self.calls: list[dict[str, Any]] = []

    def call(self, name: str, args: dict, *, user_role: str, corpora: list[str]) -> list[dict]:
        self.calls.append({"name": name, "args": args, "user_role": user_role, "corpora": corpora})
        found = self.passages(args["query"]) if callable(self.passages) else self.passages
        return list(found)
