from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

from docx.opc.exceptions import PackageNotFoundError

from ingestion.chunking import chunk_document
from ingestion.hashing import sha256_file
from ingestion.loaders import iter_supported_files, load_document


def ingest_directory(
    input_dir: str | Path,
    processed_dir: str | Path = "data/processed",
    manifest_path: str | Path = "data/manifests/ingestion.jsonl",
    chunk_strategy: str = "section",
    chunk_size: int = 1200,
    chunk_overlap: int = 180,
    source_name: str = "SGG",
    language: str = "fr",
) -> dict[str, Any]:

    input_root = Path(input_dir)

    if not input_root.exists():
        raise FileNotFoundError(input_root)

    if not input_root.is_dir():
        raise NotADirectoryError(input_root)

    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)

    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    existing_hashes = set()

    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)

                if record.get("status") == "extracted":
                    existing_hashes.add(record["sha256"])

            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    new_docs = 0
    skipped = 0
    total_chunks = 0
    errors = []

    for file_path in iter_supported_files(input_root):

        try:
            digest = sha256_file(file_path)

            if digest in existing_hashes:
                skipped += 1
                continue

            loaded = load_document(
                file_path,
                source_name=source_name,
                language=language,
            )

            chunks = chunk_document(
                loaded,
                strategy=chunk_strategy,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
            )

            document_data = {
                "metadata": loaded.metadata.model_dump(mode="json"),
                "pages": [page.model_dump(mode="json") for page in loaded.pages],
                "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
            }

            output_file = processed / (f"{loaded.metadata.document_id}.json")

            temporary_file = output_file.with_suffix(".json.tmp")

            temporary_file.write_text(
                json.dumps(
                    document_data,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            temporary_file.replace(output_file)

            record = {
                "document_id": loaded.metadata.document_id,
                "sha256": digest,
                "local_path": str(file_path.resolve()),
                "status": "extracted",
                "chunks_count": len(chunks),
            }

            with manifest.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

            existing_hashes.add(digest)
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
