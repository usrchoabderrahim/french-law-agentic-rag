from __future__ import annotations

from collections.abc import Sequence


class LocalEmbeddingModel:
    """Local sentence-transformers wrapper; no hosted embedding API is used."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "auto"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Installer l'extra retrieval : pip install -e '.[retrieval]'"
            ) from exc
        self.model = SentenceTransformer(model_name, device=None if device == "auto" else device)
        self.model_name = model_name

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()
