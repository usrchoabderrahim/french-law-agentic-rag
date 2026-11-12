from __future__ import annotations

import os
import re
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pymupdf
from bs4 import BeautifulSoup
from docx import Document as DocxDocument

from ingestion.hashing import sha256_file, stable_document_id
from ingestion.schemas import DocumentMetadata, LoadedDocument, PageText

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".markdown",
    ".html",
    ".htm",
}

# Configuration OCR française
OCR_LANGUAGE = "fra"
OCR_DPI = 300
OCR_TIMEOUT_SECONDS = 120

# Seuils de détection des pages nécessitant un OCR
MIN_EXTRACTED_TEXT_LENGTH = 100
MIN_ALPHANUMERIC_CHARACTERS = 50


def _configure_tesseract() -> None:
    """Configure the Tesseract executable on Windows or Linux."""
    import pytesseract

    configured_path = os.getenv("TESSERACT_CMD")

    if configured_path:
        executable_path = Path(configured_path)

        if not executable_path.is_file():
            raise FileNotFoundError(f"Tesseract introuvable : {configured_path}")

        pytesseract.pytesseract.tesseract_cmd = str(executable_path)
        return

    executable = shutil.which("tesseract")

    if executable:
        pytesseract.pytesseract.tesseract_cmd = executable
        return

    windows_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

    if windows_path.is_file():
        pytesseract.pytesseract.tesseract_cmd = str(windows_path)
        return

    raise FileNotFoundError(
        "Tesseract OCR introuvable. " "Vérifiez son installation ou la variable TESSERACT_CMD."
    )


def _needs_ocr(text: str) -> bool:
    """Detect PDF pages with insufficient extractable text."""
    normalized = "".join(text.split())
    alphanumeric = sum(character.isalnum() for character in normalized)

    return len(normalized) < MIN_EXTRACTED_TEXT_LENGTH or alphanumeric < MIN_ALPHANUMERIC_CHARACTERS


def _ocr_pdf_page(page) -> str:
    """Extract French text from a scanned PDF page."""
    import pytesseract
    from PIL import Image

    _configure_tesseract()

    pixmap = page.get_pixmap(
        dpi=OCR_DPI,
        alpha=False,
    )

    with Image.frombytes(
        "RGB",
        (pixmap.width, pixmap.height),
        pixmap.samples,
    ) as image:
        extracted_text = pytesseract.image_to_string(
            image,
            lang=OCR_LANGUAGE,
            config="--oem 1 --psm 3",
            timeout=OCR_TIMEOUT_SECONDS,
        )

    return extracted_text.strip()


def _parse_publication_date(raw: str | None) -> date | None:
    """Convert an ISO publication date into a Python date."""
    if not raw:
        return None

    value = raw.strip()

    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"

        return datetime.fromisoformat(value).date()

    except ValueError:
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None


def _extract_html_publication_date(
    soup: BeautifulSoup,
) -> tuple[date | None, str | None]:
    """Extract publication date from standard HTML metadata."""
    selectors = [
        'meta[property="article:published_time"]',
        'meta[property="og:published_time"]',
        'meta[itemprop="datePublished"]',
        'meta[name="date"]',
        'meta[name="dc.date"]',
        'meta[name="dcterms.date"]',
        "time[datetime]",
    ]

    raw_values = []

    for selector in selectors:
        tag = soup.select_one(selector)

        if tag is None:
            continue

        raw = tag.get("content") or tag.get("datetime")

        if not raw:
            continue

        raw = str(raw).strip()
        raw_values.append(raw)

        parsed_date = _parse_publication_date(raw)

        if parsed_date is not None:
            return parsed_date, raw

    return None, raw_values[0] if raw_values else None


def load_document(
    path: str | Path,
    source_name: str = "local_upload",
    source_url: str | None = None,
    language: str = "fr",
    metadata_overrides: dict[str, Any] | None = None,
    ocr_enabled: bool = True,
) -> LoadedDocument:
    """
    Load PDF, DOCX, TXT, Markdown and HTML documents.

    Automatically applies French OCR to PDF pages
    containing insufficient extractable text.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Format non pris en charge : {extension}")

    extra: dict[str, Any] = {}

    detected_publication_date = None
    detected_publication_date_raw = None

    # ==========================================
    # 1. PDF + OCR
    # ==========================================
    if extension == ".pdf":
        pages = []
        ocr_attempted_pages = []
        ocr_successful_pages = []

        with pymupdf.open(file_path) as pdf:
            extra = dict(pdf.metadata or {})

            for index, page in enumerate(pdf, start=1):
                extracted_text = page.get_text("text") or ""
                final_text = extracted_text

                # Détection des pages scannées
                if ocr_enabled and _needs_ocr(extracted_text):
                    ocr_attempted_pages.append(index)

                    # Reconnaissance OCR française
                    ocr_text = _ocr_pdf_page(page)

                    if ocr_text.strip():
                        final_text = ocr_text
                        ocr_successful_pages.append(index)

                pages.append(
                    PageText(
                        page_number=index,
                        text=final_text,
                    )
                )

        # Métadonnées OCR
        extra["ocr"] = {
            "enabled": ocr_enabled,
            "language": OCR_LANGUAGE,
            "dpi": OCR_DPI,
            "pages_attempted": ocr_attempted_pages,
            "pages_with_text": ocr_successful_pages,
            "pages_attempted_count": len(ocr_attempted_pages),
            "pages_with_text_count": len(ocr_successful_pages),
        }

        title = extra.get("title") or file_path.stem
        page_count = len(pages)

    # ==========================================
    # 2. DOCX
    # ==========================================
    elif extension == ".docx":
        doc = DocxDocument(file_path)

        text = "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())

        title = doc.core_properties.title or file_path.stem

        extra = {
            "author": doc.core_properties.author,
            "created": str(doc.core_properties.created or ""),
        }

        pages = [
            PageText(
                page_number=None,
                text=text,
            )
        ]

        page_count = None

    # ==========================================
    # 3. TXT / MARKDOWN
    # ==========================================
    elif extension in {
        ".txt",
        ".md",
        ".markdown",
    }:
        text = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        title = file_path.stem

        # Les marqueurs OCR sont strictement "PAGE 1", "PAGE 2", etc.
        # Ne pas ignorer la casse : "page 1159" peut être une référence
        # à une page citée dans le contenu juridique, pas un marqueur OCR.
        page_matches = list(
            re.finditer(
                r"(?m)^[ \t]*PAGE[ \t]+(\d{1,3})[ \t]*$",
                text,
            )
        )

        if page_matches:
            pages = []

            for index, match in enumerate(page_matches):
                start = match.end()

                end = (
                    page_matches[index + 1].start() if index + 1 < len(page_matches) else len(text)
                )

                page_text = text[start:end].strip()

                pages.append(
                    PageText(
                        page_number=int(match.group(1)),
                        text=page_text,
                    )
                )

            page_count = len(pages)

            extra = {
                "ocr": {
                    "enabled": True,
                    "source": "pre_extracted_ocr_text",
                    "page_markers_detected": True,
                    "pages_with_text": [page.page_number for page in pages if page.text.strip()],
                    "pages_with_text_count": sum(bool(page.text.strip()) for page in pages),
                }
            }

        else:
            pages = [
                PageText(
                    page_number=None,
                    text=text,
                )
            ]

            page_count = None

    # ==========================================
    # 4. HTML
    # ==========================================
    else:
        raw = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        soup = BeautifulSoup(
            raw,
            "html.parser",
        )

        (
            detected_publication_date,
            detected_publication_date_raw,
        ) = _extract_html_publication_date(soup)

        for node in soup(["script", "style", "noscript", "nav", "footer"]):
            node.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else file_path.stem

        text = soup.get_text(
            "\n",
            strip=True,
        )

        extra = {
            "html_title": title,
        }

        pages = [
            PageText(
                page_number=None,
                text=text,
            )
        ]

        page_count = None

    # ==========================================
    # 5. DOCUMENT IDENTIFICATION
    # ==========================================
    digest = sha256_file(file_path)
    overrides = metadata_overrides or {}

    metadata = DocumentMetadata(
        document_id=stable_document_id(digest),
        title=(overrides.get("title") or title),
        source_name=source_name,
        source_url=source_url,
        local_path=str(file_path.resolve()),
        file_type=extension.lstrip("."),
        language=language,
        publication_date=(overrides.get("publication_date") or detected_publication_date),
        publication_date_raw=(
            overrides.get("publication_date_raw") or detected_publication_date_raw
        ),
        publication_year=overrides.get("publication_year"),
        document_type=overrides.get("document_type"),
        legal_reference=overrides.get("legal_reference"),
        bo_number=overrides.get("bo_number"),
        page_count=page_count,
        sha256=digest,
        access_level=overrides.get(
            "access_level",
            "public",
        ),
        allowed_roles=overrides.get(
            "allowed_roles",
            ["reader", "admin"],
        ),
        extra={
            **extra,
            **overrides.get("extra", {}),
        },
    )

    # ==========================================
    # 6. FINAL DOCUMENT
    # ==========================================
    return LoadedDocument(
        metadata=metadata,
        pages=pages,
    )


def iter_supported_files(
    directory: str | Path,
):
    """Yield supported files recursively from a directory."""
    root = Path(directory)

    if not root.exists():
        raise FileNotFoundError(root)

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
