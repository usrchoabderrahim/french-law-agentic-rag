from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    payload = f"{document_id}\0{chunk_index}\0{text}".encode()
    return hashlib.sha256(payload).hexdigest()[:32]


def export_chunks(
    processed_dir: Path, output_file: Path, *, include_local_paths: bool = True
) -> dict[str, Any]:
    if not processed_dir.exists():
        raise FileNotFoundError(f"Dossier processed introuvable : {processed_dir}")

    processed_files = sorted(processed_dir.glob("*.json"))
    if not processed_files:
        raise RuntimeError(f"Aucun fichier JSON trouvé dans {processed_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")

    seen_ids: set[str] = set()
    exported_chunks = 0
    skipped_chunks = 0
    documents_with_chunks = 0
    errors: list[dict[str, str]] = []

    with temporary_file.open("w", encoding="utf-8", newline="\n") as output:
        for processed_file in processed_files:
            try:
                saved = json.loads(processed_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append({"file": str(processed_file), "error": str(exc)})
                continue

            document_metadata = dict(saved.get("metadata") or {})
            if not include_local_paths:
                document_metadata.pop("local_path", None)
            document_id = str(document_metadata.get("document_id") or processed_file.stem)
            chunks = saved.get("chunks") or []

            if chunks:
                documents_with_chunks += 1

            for chunk_index, chunk in enumerate(chunks):
                if not isinstance(chunk, dict):
                    skipped_chunks += 1
                    continue

                text = str(chunk.get("text") or "").strip()
                if not text:
                    skipped_chunks += 1
                    continue

                supplied_id = chunk.get("chunk_id") or chunk.get("id") or chunk.get("chunk_uuid")
                chunk_id = str(supplied_id or stable_chunk_id(document_id, chunk_index, text))

                if chunk_id in seen_ids:
                    raise ValueError(
                        "Identifiant de chunk dupliqué : "
                        f"{chunk_id}. Vérifier les fichiers dans {processed_dir}."
                    )
                seen_ids.add(chunk_id)

                # Preserve the chunk payload and its document metadata.
                row = {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "text": text,
                    "chunk_metadata": {key: value for key, value in chunk.items() if key != "text"},
                    "document_metadata": document_metadata,
                    "source_processed_file": processed_file.name,
                    "source_chunk_index": chunk_index,
                }
                output.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                exported_chunks += 1

    temporary_file.replace(output_file)
    file_hash = sha256_file(output_file)

    manifest = {
        "format_version": 1,
        "status": "pending_embeddings",
        "embedding_model": "BAAI/bge-m3",
        "expected_embedding_dimension": 1024,
        "language_scope": ["fr"],
        "processed_files_found": len(processed_files),
        "documents_with_chunks": documents_with_chunks,
        "chunks_exported": exported_chunks,
        "chunks_skipped_empty_or_invalid": skipped_chunks,
        "input_file": output_file.name,
        "input_sha256": file_hash,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "errors": errors,
    }
    if include_local_paths:
        manifest["processed_directory"] = str(processed_dir.resolve())

    manifest_path = output_file.with_name("chunk_export_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exporter les chunks préparés vers un JSONL compatible Colab."
    )
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument(
        "--output",
        default="data/colab_input/all_chunks.jsonl",
    )
    args = parser.parse_args()

    manifest = export_chunks(Path(args.processed), Path(args.output))
    print("\n========== EXPORT POUR GOOGLE COLAB ==========")
    print(f"Documents traités : {manifest['processed_files_found']}")
    print(f"Documents avec chunks : {manifest['documents_with_chunks']}")
    print(f"Chunks exportés : {manifest['chunks_exported']}")
    print(f"Chunks ignorés : {manifest['chunks_skipped_empty_or_invalid']}")
    print(f"JSONL : {Path(args.output).resolve()}")
    print("Manifest : " f"{Path(args.output).with_name('chunk_export_manifest.json').resolve()}")
    print(f"SHA-256 : {manifest['input_sha256']}")
    if manifest["errors"]:
        print(f"Fichiers en erreur : {len(manifest['errors'])}")


if __name__ == "__main__":
    main()
