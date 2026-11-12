from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from ingestion.config import load_config
from ingestion.discovery import discover_sgg_pdf_links
from ingestion.download import append_jsonl, download_pdf
from ingestion.hashing import stable_document_id


def main():
    parser = argparse.ArgumentParser(
        description=("Découvrir et télécharger les Bulletins officiels " "français du SGG.")
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Découverte sans téléchargement.",
    )

    mode.add_argument(
        "--download",
        action="store_true",
        help="Télécharger les candidats validés.",
    )

    parser.add_argument("--config", default="config.yaml")

    args = parser.parse_args()

    config = load_config(args.config)

    collection = config["collection"]
    source = config["sources"]["sgg"]

    root = Path(collection["data_dir"])

    catalogue_dir = root / "catalogues" / "sgg"
    raw_dir = root / "raw" / "sgg_bo_fr"
    reports_dir = root / "reports"
    manifests_dir = root / "manifests"

    for folder in (
        catalogue_dir,
        raw_dir,
        reports_dir,
        manifests_dir,
    ):
        folder.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")

    final_url, candidates, html = discover_sgg_pdf_links(
        catalogue_url=source["catalogue_url"],
        allowed_hosts=source["allowed_hosts"],
        recent_years=collection["recent_years"],
        user_agent=collection["user_agent"],
        timeout=collection["request_timeout_seconds"],
        delay=collection["request_delay_seconds"],
    )

    catalogue_file = catalogue_dir / f"{run_id}_catalogue.html"

    catalogue_file.write_text(
        html,
        encoding="utf-8",
    )

    report_path = reports_dir / f"{run_id}_sgg_candidates.csv"

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "url",
                "label",
                "source_page",
                "detected_year",
                "language",
                "review_reason",
            ],
        )

        writer.writeheader()

        writer.writerows([candidate.model_dump() for candidate in candidates])

    print(f"Catalogue analysé : {final_url}")
    print(f"Candidats PDF : {len(candidates)}")
    print(f"Rapport : {report_path}")

    if not args.download:

        print("Simulation : aucun PDF téléchargé. " "Inspecter le rapport avant --download.")

        for candidate in candidates[:25]:

            print(
                f"- année={candidate.detected_year} "
                f"langue={candidate.language} "
                f"revue={bool(candidate.review_reason)} "
                f"| {candidate.url}"
            )

        return

    manifest_path = manifests_dir / "sgg_bo_fr_manifest.jsonl"

    previous_records = []

    if manifest_path.exists():

        for line in manifest_path.read_text(encoding="utf-8").splitlines():

            try:
                previous_records.append(json.loads(line))

            except json.JSONDecodeError:
                continue

    seen_urls = {row.get("url") for row in previous_records if row.get("url")}

    seen_hashes = {
        row.get("sha256"): (row.get("document_id") or stable_document_id(row["sha256"]))
        for row in previous_records
        if row.get("sha256")
    }

    for candidate in candidates:

        if candidate.url in seen_urls:

            print("unchanged_url_skipped", candidate.url)
            continue

        result = download_pdf(
            candidate=candidate,
            output_dir=raw_dir,
            allowed_hosts=source["allowed_hosts"],
            user_agent=collection["user_agent"],
            timeout=collection["request_timeout_seconds"],
            delay=collection["request_delay_seconds"],
            max_size_mb=collection["max_pdf_size_mb"],
        )

        if result.get("status") == "downloaded":

            digest = result["sha256"]

            document_id = stable_document_id(digest)

            if digest in seen_hashes:

                duplicate_path = result.get("local_path")

                if duplicate_path:
                    Path(duplicate_path).unlink(missing_ok=True)

                result["status"] = "duplicate_content"
                result["duplicate_of"] = seen_hashes[digest]
                result["local_path"] = None

            else:

                result["document_id"] = document_id
                seen_hashes[digest] = document_id

        result["run_id"] = run_id

        result["collected_at"] = datetime.now(UTC).isoformat()

        result["source_name"] = source["name"]

        append_jsonl(manifest_path, result)

        seen_urls.add(candidate.url)

        print(
            result["status"],
            result.get("url", ""),
        )

    print(f"Manifeste : {manifest_path}")


if __name__ == "__main__":
    main()
