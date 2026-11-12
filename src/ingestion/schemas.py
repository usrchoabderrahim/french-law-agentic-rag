from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class SourceCandidate(BaseModel):
    url: str
    label: str = ""
    source_page: str
    detected_year: int | None = None
    language: str | None = None
    review_reason: str | None = None


class DocumentMetadata(BaseModel):
    document_id: str
    title: str
    source_name: str
    source_url: str | None = None
    local_path: str
    file_type: str
    language: str = "fr"
    publication_date: date | None = None
    publication_date_raw: str | None = None
    publication_year: int | None = None
    document_type: str | None = None
    legal_reference: str | None = None
    bo_number: str | None = None
    page_count: int | None = None
    sha256: str
    access_level: str = "public"
    allowed_roles: list[str] = Field(default_factory=lambda: ["reader", "admin"])
    extra: dict[str, Any] = Field(default_factory=dict)


class PageText(BaseModel):
    page_number: int | None = None
    text: str
    section: str | None = None


class LoadedDocument(BaseModel):
    metadata: DocumentMetadata
    pages: list[PageText]


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
