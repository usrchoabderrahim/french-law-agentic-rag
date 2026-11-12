from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from export_chunks_for_colab import export_chunks

from ingestion.config import load_config
from ingestion.hashing import sha256_file, stable_document_id
from ingestion.pipeline import ingest_directory

SOURCES = {
    "fstbm": {
        "normes_complementaires_lst.pdf": (
            "https://fstbm.ac.ma/web/content/1489951/Normes_compl%C3%A9mentaires_CNPN-LST.pdf",
            "Normes pédagogiques complémentaires de la licence en sciences et techniques",
            {"document_type": "normes_pedagogiques"},
        ),
        "charte_these.pdf": (
            "https://fstbm.ac.ma/web/content/1417391?download=true",
            "Charte de thèse de doctorat",
            {"document_type": "charte"},
        ),
        "futurs_etudiants.html": (
            "https://fstbm.ac.ma/futurs-etudiants",
            "Futurs étudiants",
            {"document_type": "page_information"},
        ),
        "guide_e_scolarite.html": (
            (
                "https://fstbm.ac.ma/blog/blog-de-l-espace-etudiant-1/post/"
                "comment-s-inscrire-sur-la-plateforme-e-scolarite-1"
            ),
            "Comment s’inscrire sur la plateforme E-Scolarité ?",
            {"document_type": "guide"},
        ),
    },
    "archives_ocr": {
        "bo_2240_1955.pdf": (
            "https://www.sgg.gov.ma/BO/bo_fr/1955/bo_2240_fr.pdf",
            "Bulletin officiel n° 2240 du 30 septembre 1955",
            {"bo_number": "2240", "publication_date": "1955-09-30", "publication_year": 1955},
        ),
        "bo_4478_1997.pdf": (
            "https://www.sgg.gov.ma/BO/bo_fr/1997/bo_4478_fr.pdf",
            "Bulletin officiel n° 4478 du 1er mai 1997",
            {"bo_number": "4478", "publication_date": "1997-05-01", "publication_year": 1997},
        ),
        "bo_5236_2004.pdf": (
            "https://www.sgg.gov.ma/BO/bo_fr/2004/bo_5236_fr.pdf",
            "Bulletin officiel n° 5236 du 5 août 2004",
            {"bo_number": "5236", "publication_date": "2004-08-05", "publication_year": 2004},
        ),
    },
}


def prepare_group(
    group: str, raw_root: Path, output_root: Path, chunking: dict, *, reprocess: bool
) -> tuple[Path, int]:
    input_dir = raw_root / group
    benchmark_only = group == "archives_ocr"
    records = {}
    digests = {}
    for filename, (url, title, metadata) in SOURCES[group].items():
        path = input_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Source manquante : {path}\nTéléchargement : {url}")
        digests[filename] = sha256_file(path)
        records[filename] = {
            "source_name": "SGG" if benchmark_only else "FST Béni Mellal",
            "source_url": url,
            "language": "fr",
            "metadata_overrides": {
                "title": title,
                **({"document_type": "bulletin_officiel"} if benchmark_only else {}),
                **metadata,
                "extra": {"corpus": group, "benchmark_only": benchmark_only},
            },
        }

    document_ids = {stable_document_id(digest) for digest in digests.values()}
    if len(document_ids) != len(records):
        raise ValueError("Deux sources du lot ont le même contenu.")

    # A changed source set gets its own document directory, so an old edition
    # cannot be picked up by the exporter after one of its files is replaced.
    snapshot = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()[:16]
    run_dir = output_root / group / snapshot
    documents = run_dir / "documents"
    ingestion_manifest = run_dir / "ingestion_manifest.jsonl"
    result = ingest_directory(
        input_dir,
        processed_dir=documents,
        manifest_path=ingestion_manifest,
        chunk_strategy=chunking["strategy"],
        chunk_size=chunking["chunk_size"],
        chunk_overlap=chunking["chunk_overlap"],
        source_records=records,
        reprocess=reprocess,
    )
    if result["errors"]:
        details = "\n".join(
            f"- {Path(error['file']).name}: {error['error']}" for error in result["errors"]
        )
        raise RuntimeError(f"Ingestion incomplète. Journal : {ingestion_manifest}\n{details}")
    if {path.stem for path in documents.glob("*.json")} != document_ids:
        raise RuntimeError(
            f"Les documents produits ne correspondent pas au lot attendu : {documents}"
        )

    archive_path = output_root / f"{group}_chunks_colab.zip"
    with TemporaryDirectory(dir=run_dir) as temporary:
        export_dir = Path(temporary)
        chunks_file = export_dir / "all_chunks.jsonl"
        manifest = export_chunks(documents, chunks_file, include_local_paths=False)
        if (
            manifest["errors"]
            or manifest["chunks_skipped_empty_or_invalid"]
            or manifest["documents_with_chunks"] != len(records)
        ):
            raise RuntimeError(f"Export incomplet : {documents}")
        manifest_path = export_dir / "chunk_export_manifest.json"
        manifest.update({"corpus": group, "benchmark_only": benchmark_only})
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        staged_archive = export_dir / archive_path.name
        with ZipFile(staged_archive, "w", compression=ZIP_DEFLATED) as archive:
            archive.write(chunks_file, chunks_file.name)
            archive.write(manifest_path, manifest_path.name)
        staged_archive.replace(archive_path)

    return archive_path, manifest["chunks_exported"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Préparer les documents publics de la FSTBM et les archives OCR du projet."
    )
    parser.add_argument("--group", choices=("all", *SOURCES), default="all")
    parser.add_argument("--input", type=Path, default=Path("data/raw/fstbm_pilot"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/fstbm_corpus"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--reprocess", action="store_true", help="Retraiter les fichiers déjà ingérés."
    )
    args = parser.parse_args()

    chunking = load_config(args.config)["ingestion"]["chunking"]
    groups = SOURCES if args.group == "all" else (args.group,)
    failed = False
    for group in groups:
        print(f"Préparation du lot {group} ({len(SOURCES[group])} documents)…", flush=True)
        try:
            archive, count = prepare_group(
                group, args.input, args.output, chunking, reprocess=args.reprocess
            )
        except (OSError, RuntimeError, ValueError) as error:
            failed = True
            print(f"{group} : {error}", file=sys.stderr)
            continue
        print(f"{group} : {count} chunks · {archive.resolve()}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
