from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

from docx.opc.exceptions import PackageNotFoundError

from ingestion.chunking import chunk_document
from ingestion.download import append_jsonl
from ingestion.hashing import sha256_file, stable_document_id
from ingestion.loaders import OCR_DPI, OCR_LANGUAGE, iter_supported_files, load_document


def ingest_directory(
    input_dir: str | Path,
    processed_dir: str | Path = "data/processed",
    manifest_path: str | Path = "data/manifests/ingestion.jsonl",
    chunk_strategy: str = "section",
    chunk_size: int = 1200,
    chunk_overlap: int = 180,
    source_name: str = "SGG",
    language: str = "fr",
    *,
    source_records: Mapping[str, dict[str, Any]] | None = None,
    reprocess: bool = False,
) -> dict[str, Any]:
    """Ingest selected files and record incomplete OCR without publishing its chunks."""
    input_root = Path(input_dir)
    if not input_root.exists():
        raise FileNotFoundError(input_root)
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)

    files = {
        path.relative_to(input_root).as_posix(): path for path in iter_supported_files(input_root)
    }
    if source_records is not None:
        missing = source_records.keys() - files.keys()
        if missing:
            raise ValueError(
                f"Sources absentes ou non prises en charge : {', '.join(sorted(missing))}"
            )
        files = {name: files[name] for name in source_records}

    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    manifest = Path(manifest_path)

    latest_records = {}
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                latest_records[record["sha256"]] = record
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    new_docs = 0
    skipped = 0
    total_chunks = 0
    errors = []
    processed_this_run = {}
    chunking_options = {
        "strategy": chunk_strategy,
        "chunk_size": chunk_size,
        "overlap": chunk_overlap,
    }

    for name, file_path in files.items():
        try:
            source = source_records[name] if source_records is not None else {}
            source_options = {
                "source_name": source.get("source_name", source_name),
                "source_url": source.get("source_url"),
                "language": source.get("language", language),
                "metadata_overrides": source.get("metadata_overrides"),
            }
            signature_data = {
                "source": source_options,
                "format": file_path.suffix.lower(),
                "chunking": chunking_options,
                "ocr": {"language": OCR_LANGUAGE, "dpi": OCR_DPI},
            }
            signature = sha256(
                json.dumps(signature_data, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            digest = sha256_file(file_path)
            if digest in processed_this_run and processed_this_run[digest] != signature:
                raise ValueError(
                    "Deux sources de contenu identique ont des métadonnées différentes."
                )

            document_id = stable_document_id(digest)
            output_file = processed / f"{document_id}.json"
            previous = latest_records.get(digest, {})
            if digest in processed_this_run or (
                not reprocess
                and previous.get("status") == "extracted"
                and previous.get("processing_signature") == signature
                and output_file.is_file()
            ):
                processed_this_run[digest] = signature
                skipped += 1
                continue

            loaded = load_document(file_path, **source_options)
            if loaded.metadata.sha256 != digest:
                raise ValueError("Le document a changé pendant son extraction.")

            record = {
                "document_id": document_id,
                "sha256": digest,
                "local_path": str(file_path.resolve()),
                "source_url": loaded.metadata.source_url,
                "processing_signature": signature,
            }
            ocr = loaded.metadata.extra.get("ocr", {})
            if failed_pages := ocr.get("pages_failed"):
                failure = {**record, "status": "ocr_incomplete", "ocr": ocr}
                append_jsonl(manifest, failure)
                latest_records[digest] = failure
                errors.append(
                    {
                        "file": str(file_path),
                        "error": "OCR incomplet sur les pages " + ", ".join(map(str, failed_pages)),
                        "error_type": "OCRIncomplete",
                        "pages_failed": failed_pages,
                    }
                )
                continue

            chunks = chunk_document(loaded, **chunking_options)
            if not chunks:
                raise ValueError("Aucun texte exploitable à découper.")
            document_data = {
                "metadata": loaded.metadata.model_dump(mode="json"),
                "pages": [page.model_dump(mode="json") for page in loaded.pages],
                "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
            }
            temporary_file = output_file.with_suffix(".json.tmp")
            temporary_file.write_text(
                json.dumps(document_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary_file.replace(output_file)

            success = {**record, "status": "extracted", "chunks_count": len(chunks)}
            append_jsonl(manifest, success)
            latest_records[digest] = success
            processed_this_run[digest] = signature
            new_docs += 1
            total_chunks += len(chunks)

        except (
            OSError,
            ValueError,
            RuntimeError,
            BadZipFile,
            PackageNotFoundError,
        ) as exc:
            errors.append(
                {
                    "file": str(file_path),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )

    return {
        "new_documents": new_docs,
        "skipped_unchanged": skipped,
        "chunks_created": total_chunks,
        "errors": errors,
        "processed_dir": str(processed.resolve()),
    }
