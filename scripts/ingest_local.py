from __future__ import annotations

import argparse

from ingestion.config import load_config
from ingestion.pipeline import ingest_directory


def main():
    parser = argparse.ArgumentParser(description="Importer un dossier documentaire local.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    result = ingest_directory(
        input_dir=args.input,
        processed_dir="data/processed",
        manifest_path="data/manifests/ingestion.jsonl",
        chunk_strategy=config["ingestion"]["chunking"]["strategy"],
        chunk_size=config["ingestion"]["chunking"]["chunk_size"],
        chunk_overlap=config["ingestion"]["chunking"]["chunk_overlap"],
    )
    print(result)


if __name__ == "__main__":
    main()
