"""Encode user questions with the SAME model used for the documents (bge-m3)."""
from __future__ import annotations

from .config import EMBEDDING_MODEL


class QueryEncoder:
    """Lazy local wrapper around sentence-transformers (no hosted API)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name, device=None if self.device == "auto" else self.device
            )
        return self._model

    def encode(self, text: str) -> list[float]:
        vector = self._load().encode(
            [text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0]
        return vector.tolist()
