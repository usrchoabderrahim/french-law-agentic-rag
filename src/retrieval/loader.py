"""Load chunks and their precomputed embeddings, with strict validation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import CORPORA, EMBEDDING_DIM


@dataclass
class IndexedChunk:
    """A chunk ready to be indexed (text + vector + flat metadata)."""

    chunk_id: str
    corpus: str
    text: str
    embedding: np.ndarray
    document_id: str
    title: str
    source_name: str
    source_url: str | None
    document_type: str | None
    language: str
    publication_year: int | None
    page_start: int | None
    page_end: int | None
    section: str | None
    article_number: str | None
    allowed_roles: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def normalize_roles(roles: list[str]) -> list[str]:
    """Keep the roles exactly as stored in the data (sorted, no duplicates)."""
    return sorted(set(roles))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_corpus(name: str, folder: Path | None = None) -> list[IndexedChunk]:
    """Load one corpus and verify chunk/vector alignment."""
    folder = folder or CORPORA[name]
    chunks = _read_jsonl(folder / "all_chunks.jsonl")
    ids = _read_jsonl(folder / "embedding_ids.jsonl")
    vectors = np.load(folder / "embeddings.npy")

    if vectors.shape != (len(chunks), EMBEDDING_DIM):
        raise ValueError(f"{name}: bad matrix shape {vectors.shape}")
    if len(ids) != len(chunks):
        raise ValueError(f"{name}: {len(ids)} ids for {len(chunks)} chunks")

    result: list[IndexedChunk] = []
    for row, (chunk, id_row) in enumerate(zip(chunks, ids)):
        if chunk["chunk_id"] != id_row["chunk_id"] or id_row["row_index"] != row:
            raise ValueError(f"{name}: chunk/vector misalignment at row {row}")

        cmeta = chunk["chunk_metadata"]
        inner = cmeta.get("metadata", {})
        dmeta = chunk["document_metadata"]
        result.append(
            IndexedChunk(
                chunk_id=chunk["chunk_id"],
                corpus=name,
                text=chunk["text"],
                embedding=vectors[row],
                document_id=chunk["document_id"],
                title=dmeta.get("title") or inner.get("title") or "",
                source_name=dmeta.get("source_name") or "",
                source_url=dmeta.get("source_url"),
                document_type=dmeta.get("document_type"),
                language=dmeta.get("language") or "fr",
                publication_year=dmeta.get("publication_year"),
                page_start=cmeta.get("page_start"),
                page_end=cmeta.get("page_end"),
                section=cmeta.get("section"),
                article_number=inner.get("article_number"),
                allowed_roles=normalize_roles(dmeta.get("allowed_roles") or []),
                extra={"act_title": inner.get("act_title"), "bo_number": dmeta.get("bo_number")},
            )
        )
    return result


def load_all() -> dict[str, list[IndexedChunk]]:
    """Load every configured corpus; corpora stay separate (no merge)."""
    return {name: load_corpus(name) for name in CORPORA}










































# """Load chunks and their precomputed embeddings, with strict validation."""
# from __future__ import annotations

# import json
# from dataclasses import dataclass, field
# from pathlib import Path
# from typing import Any

# import numpy as np

# from .config import CORPORA, EMBEDDING_DIM, ROLE_ALIASES


# @dataclass
# class IndexedChunk:
#     """A chunk ready to be indexed (text + vector + flat metadata)."""

#     chunk_id: str
#     corpus: str
#     text: str
#     embedding: np.ndarray
#     document_id: str
#     title: str
#     source_name: str
#     source_url: str | None
#     document_type: str | None
#     language: str
#     publication_year: int | None
#     page_start: int | None
#     page_end: int | None
#     section: str | None
#     article_number: str | None
#     allowed_roles: list[str] = field(default_factory=list)
#     extra: dict[str, Any] = field(default_factory=dict)


# def normalize_roles(roles: list[str]) -> list[str]:
#     """Map ingestion role names to spec role names; admin is always allowed."""
#     result = {ROLE_ALIASES.get(role, role) for role in roles}
#     result.add("admin")
#     return sorted(result)


# def _read_jsonl(path: Path) -> list[dict[str, Any]]:
#     with path.open("r", encoding="utf-8") as stream:
#         return [json.loads(line) for line in stream if line.strip()]


# def load_corpus(name: str, folder: Path | None = None) -> list[IndexedChunk]:
#     """Load one corpus and verify chunk/vector alignment."""
#     folder = folder or CORPORA[name]
#     chunks = _read_jsonl(folder / "all_chunks.jsonl")
#     ids = _read_jsonl(folder / "embedding_ids.jsonl")
#     vectors = np.load(folder / "embeddings.npy")

#     if vectors.shape != (len(chunks), EMBEDDING_DIM):
#         raise ValueError(f"{name}: bad matrix shape {vectors.shape}")
#     if len(ids) != len(chunks):
#         raise ValueError(f"{name}: {len(ids)} ids for {len(chunks)} chunks")

#     result: list[IndexedChunk] = []
#     for row, (chunk, id_row) in enumerate(zip(chunks, ids)):
#         if chunk["chunk_id"] != id_row["chunk_id"] or id_row["row_index"] != row:
#             raise ValueError(f"{name}: chunk/vector misalignment at row {row}")

#         cmeta = chunk["chunk_metadata"]
#         inner = cmeta.get("metadata", {})
#         dmeta = chunk["document_metadata"]
#         result.append(
#             IndexedChunk(
#                 chunk_id=chunk["chunk_id"],
#                 corpus=name,
#                 text=chunk["text"],
#                 embedding=vectors[row],
#                 document_id=chunk["document_id"],
#                 title=dmeta.get("title") or inner.get("title") or "",
#                 source_name=dmeta.get("source_name") or "",
#                 source_url=dmeta.get("source_url"),
#                 document_type=dmeta.get("document_type"),
#                 language=dmeta.get("language") or "fr",
#                 publication_year=dmeta.get("publication_year"),
#                 page_start=cmeta.get("page_start"),
#                 page_end=cmeta.get("page_end"),
#                 section=cmeta.get("section"),
#                 article_number=inner.get("article_number"),
#                 allowed_roles=normalize_roles(dmeta.get("allowed_roles") or []),
#                 extra={"act_title": inner.get("act_title"), "bo_number": dmeta.get("bo_number")},
#             )
#         )
#     return result


# def load_all() -> dict[str, list[IndexedChunk]]:
#     """Load every configured corpus; corpora stay separate (no merge)."""
#     return {name: load_corpus(name) for name in CORPORA}
