"""Outils de texte partagés par les nœuds (normalisation, mots-clés, phrases)."""
from __future__ import annotations

import re
import unicodedata

from src.retrieval.bm25 import tokenize  # même normalisation que la recherche lexicale

_EXTRA_STOP = {"quel", "quelle", "quels", "quelles", "comment", "combien", "pourquoi", "quand",
               "doit", "peut", "faut", "selon", "propos", "etre", "avoir", "fait", "lors"}


def fold(text: str) -> str:
    """minuscules, sans accents, apostrophes typographiques unifiées, espaces réduits."""
    text = unicodedata.normalize("NFKD", text.replace("’", "'"))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    return re.sub(r"\s+", " ", text).strip()


def stem(token: str) -> str:
    return token[:6] if not token.isdigit() else token


def keywords(text: str) -> set[str]:
    """Mots-clés racinisés (6 premières lettres) hors mots vides."""
    return {stem(t) for t in tokenize(text) if t not in _EXTRA_STOP and len(t) > 2 or t.isdigit()}


def sentences(text: str, max_len: int = 400) -> list[str]:
    parts = re.split(r"(?<=[.;:!?])\s+|\n+", text)
    return [p.strip()[:max_len] for p in parts if len(p.strip()) >= 15]
