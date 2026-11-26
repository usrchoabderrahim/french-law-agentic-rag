"""Couche LLM des agents : client Ollama local + moteur hors-ligne déterministe.

Tous les agents passent par `ask_json` qui :
  * construit l'invite (TASK + consignes + DATA JSON),
  * compte les appels et les tokens dans la trace, applique le budget global,
  * extrait le JSON de la réponse et réessaie si le JSON est invalide.

Backends (variable d'environnement RAG_LLM) :
  * "ollama"     : LLM local via Ollama (mêmes réglages que advanced_config.py) ;
  * "openrouter" : API OpenRouter, modèles gratuits (OPENROUTER_API_KEY, OPENROUTER_MODEL) ;
  * "heuristic"  : moteur hors-ligne déterministe. Chaque nœud enregistre sa propre
                   heuristique avec `@register_heuristic("<tâche>")`. Il sert aux démos
                   sans GPU et rend le comportement du graphe reproductible.
  * "auto" (défaut) : Ollama s'il répond, sinon OpenRouter si la clé existe, sinon heuristic.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Protocol

from . import config
from .state import AgentState


class LLMUnavailable(RuntimeError):
    """Le serveur LLM ne répond pas."""


class BudgetExceeded(RuntimeError):
    """Budget global (appels LLM ou latence) dépassé : le graphe part en repli."""


class LLMClient(Protocol):
    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str: ...


# --------------------------------------------------------------------------- #
# Backend 1 : Ollama (local, aucune API hébergée)
# --------------------------------------------------------------------------- #
class OllamaLLM:
    def __init__(self, model: str | None = None, base_url: str | None = None, timeout: int = 180):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.base_url = (base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3) as resp:
                names = {m.get("name", "") for m in json.loads(resp.read()).get("models", [])}
        except (urllib.error.URLError, OSError, ValueError):
            return False
        return self.model in names or f"{self.model}:latest" in names

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.1},
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"
        request = urllib.request.Request(
            f"{self.base_url}/api/generate", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                return (json.loads(resp.read()).get("response") or "").strip()
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise LLMUnavailable(f"Ollama injoignable ({self.base_url}, {self.model}): {exc}") from exc


# --------------------------------------------------------------------------- #
# Backend 2 : OpenRouter (API compatible OpenAI, modèles gratuits « :free »)
# --------------------------------------------------------------------------- #
class OpenRouterLLM:
    """La clé est lue dans OPENROUTER_API_KEY (jamais écrite dans un fichier ni journalisée).

    OPENROUTER_MODEL accepte une liste séparée par des virgules : quand un modèle gratuit
    est saturé (HTTP 429), on passe au suivant.
    """

    URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MODELS = ("nvidia/nemotron-3-ultra-550b-a55b:free,nvidia/nemotron-3.5-super-vl:free,"
                      "google/gemma-4-31b-it:free")

    def __init__(self, model: str | None = None, timeout: int = 120, max_retries: int = 2):
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        names = model or os.getenv("OPENROUTER_MODEL") or self.DEFAULT_MODELS
        self.models = [m.strip() for m in names.split(",") if m.strip()]
        self.model = self.models[0]
        self.timeout = timeout
        self.max_retries = max_retries

    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        if not self.api_key:
            raise LLMUnavailable("OPENROUTER_API_KEY absente")
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                   "X-Title": "french-law-agentic-rag"}
        last_error = ""
        for attempt in range(self.max_retries + 1):
            for model in self.models:            # modèle saturé → modèle suivant
                body = json.dumps({"model": model, "messages": messages,
                                   "temperature": 0.1}).encode()
                request = urllib.request.Request(self.URL, data=body, headers=headers)
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                        data = json.loads(resp.read())
                    self.model = model
                    return (data["choices"][0]["message"].get("content") or "").strip()
                except urllib.error.HTTPError as exc:
                    if exc.code not in (429, 502, 503):
                        raise LLMUnavailable(f"OpenRouter HTTP {exc.code} ({model})") from exc
                    last_error = f"HTTP {exc.code} ({model})"
                except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as exc:
                    raise LLMUnavailable(f"OpenRouter injoignable ({model}): {exc}") from exc
            if attempt < self.max_retries:
                time.sleep(5 * (attempt + 1))    # tous saturés : petite pause puis nouvel essai
        raise LLMUnavailable(f"OpenRouter : limite de débit persistante, dernier {last_error}")


# --------------------------------------------------------------------------- #
# Backend 3 : moteur hors-ligne déterministe (heuristiques par tâche)
# --------------------------------------------------------------------------- #
HEURISTICS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}


def register_heuristic(task: str):
    def deco(fn: Callable[[dict[str, Any]], dict[str, Any]]):
        HEURISTICS[task] = fn
        return fn
    return deco


_TASK_RE = re.compile(r"^TASK:\s*(\S+)", re.M)
_DATA_RE = re.compile(r"^DATA:\s*\n(.*)\Z", re.S | re.M)


class HeuristicLLM:
    """Répond aux invites construites par `build_prompt` avec l'heuristique de la tâche."""

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        task, data = _TASK_RE.search(prompt), _DATA_RE.search(prompt)
        if not task or not data or task.group(1) not in HEURISTICS:
            return "{}"
        return json.dumps(HEURISTICS[task.group(1)](json.loads(data.group(1))), ensure_ascii=False)


def get_llm(kind: str | None = None) -> LLMClient:
    kind = (kind or os.getenv("RAG_LLM", "auto")).lower()
    if kind == "heuristic":
        return HeuristicLLM()
    if kind == "openrouter":
        return OpenRouterLLM()
    ollama = OllamaLLM()
    if kind == "ollama" or ollama.available():
        return ollama
    if OpenRouterLLM().available():
        return OpenRouterLLM()
    return HeuristicLLM()


# --------------------------------------------------------------------------- #
# Appel JSON commun à tous les agents
# --------------------------------------------------------------------------- #
def build_prompt(task: str, instructions: str, data: dict[str, Any]) -> str:
    return (
        f"TASK: {task}\n{instructions.strip()}\n"
        "Réponds uniquement par un objet JSON valide.\n"
        f"DATA:\n{json.dumps(data, ensure_ascii=False)}"
    )


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Premier objet JSON de la réponse (tolère du texte ou des ``` autour)."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                try:
                    value = json.loads(text[start : i + 1])
                    return value if isinstance(value, dict) else None
                except ValueError:
                    break
        start = text.find("{", start + 1)
    return None


def check_budget(state: AgentState) -> None:
    trace = state["trace"]
    if trace["llm_calls"] >= config.MAX_LLM_CALLS:
        raise BudgetExceeded(f"{trace['llm_calls']} appels LLM")
    if time.time() - trace["started_at"] > config.MAX_LATENCY_SECONDS:
        raise BudgetExceeded("latence maximale dépassée")


def ask_json(
    state: AgentState,
    llm: LLMClient,
    task: str,
    system: str,
    instructions: str,
    data: dict[str, Any],
    retries: int = 1,
    validate: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any] | None:
    """Appelle le LLM et renvoie un dict valide, ou None après `retries` nouveaux essais."""
    prompt = build_prompt(task, instructions, data)
    for _ in range(retries + 1):
        check_budget(state)
        raw = llm.generate(prompt, system=system, json_mode=True)
        trace = state["trace"]
        trace["llm_calls"] += 1
        trace["tokens"] += (len(prompt) + len(system) + len(raw)) // 4  # estimation
        parsed = parse_json_object(raw)
        if parsed is not None and (validate is None or validate(parsed)):
            return parsed
    return None
