"""Client minimal pour un LLM local Ollama (aucune API hébergée, aucune dépendance en plus)."""
from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

from .advanced_config import (
    LLM_CACHE_PATH,
    LLM_TEMPERATURE,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
)


class LLMUnavailable(RuntimeError):
    """Ollama ne répond pas (serveur éteint, modèle absent, timeout...)."""


class LLM(Protocol):
    """Tout objet avec cette méthode convient (OllamaClient, faux LLM dans les tests)."""

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str: ...


class OllamaClient:
    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_URL,
        timeout: int = OLLAMA_TIMEOUT,
        temperature: float = LLM_TEMPERATURE,
        cache_path: Path | None = LLM_CACHE_PATH,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, str] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    # -- état du serveur ------------------------------------------------------
    def available(self) -> bool:
        """True si le serveur répond ET que le modèle est installé."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=5) as resp:
                tags = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
            return False
        names = {m.get("name", "") for m in tags.get("models", [])}
        return self.model in names or f"{self.model}:latest" in names

    # -- génération -------------------------------------------------------------
    def _key(self, prompt: str, system: str | None, json_mode: bool) -> str:
        raw = json.dumps([self.model, self.temperature, system, prompt, json_mode], ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        key = self._key(prompt, system, json_mode)
        if key in self._cache:
            return self._cache[key]

        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError) as exc:
            raise LLMUnavailable(
                f"Ollama injoignable sur {self.base_url} (modèle '{self.model}'): {exc}"
            ) from exc

        text = (body.get("response") or "").strip()
        if not text:
            raise LLMUnavailable("Réponse vide d'Ollama")
        self._cache[key] = text
        self._save_cache()
        return text

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.cache_path)
