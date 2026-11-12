from __future__ import annotations

import argparse

from ingestion.config import load_config
from ingestion.pipeline import ingest_directory


def main() -> int:
    parser = argparse.ArgumentParser(description="Importer un dossier documentaire local.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--manifest", default="data/manifests/ingestion.jsonl")
    parser.add_argument("--source-name", default="SGG")
    parser.add_argument("--reprocess", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    result = ingest_directory(
        input_dir=args.input,
        processed_dir=args.processed,
        manifest_path=args.manifest,
        chunk_strategy=config["ingestion"]["chunking"]["strategy"],
        chunk_size=config["ingestion"]["chunking"]["chunk_size"],
        chunk_overlap=config["ingestion"]["chunking"]["chunk_overlap"],
        source_name=args.source_name,
        reprocess=args.reprocess,
    )
    print(result)
    return int(bool(result["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
