import json

import pymupdf

from ingestion import loaders
from ingestion.pipeline import ingest_directory


def test_ingestion_reports_failed_ocr_pages_and_retries(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    scanned = raw / "reglement.pdf"
    with pymupdf.open() as image_source, pymupdf.open() as document:
        source_page = image_source.new_page()
        source_page.insert_text((40, 40), "Page numerisee du reglement.")
        native_page = document.new_page()
        native_page.insert_text((40, 40), "Page native du reglement.")
        scan_page = document.new_page()
        scan_page.insert_image(scan_page.rect, pixmap=source_page.get_pixmap())
        document.save(scanned)

    def fail_ocr(_page):
        raise RuntimeError("OCR timeout")

    monkeypatch.setattr(loaders, "_ocr_pdf_page", fail_ocr)
    processed = tmp_path / "processed"
    manifest = tmp_path / "ingestion.jsonl"
    result = ingest_directory(raw, processed, manifest)

    assert result["new_documents"] == 0
    assert result["errors"][0]["pages_failed"] == [2]
    assert not list(processed.glob("*.json"))
    failure = json.loads(manifest.read_text(encoding="utf-8"))
    assert failure["status"] == "ocr_incomplete"
    assert failure["ocr"]["page_diagnostics"][1]["source"] == "failed"

    monkeypatch.setattr(loaders, "_ocr_pdf_page", lambda _page: "Page OCR du règlement.")
    result = ingest_directory(raw, processed, manifest)

    assert result["errors"] == []
    assert result["new_documents"] == 1
    output = json.loads(next(processed.glob("*.json")).read_text(encoding="utf-8"))
    assert output["metadata"]["extra"]["ocr"]["pages_failed"] == []
    assert [page["page_number"] for page in output["pages"]] == [1, 2]


def test_ingestion_reuses_only_matching_source_and_settings(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "guide.txt").write_text("Informations sur les études.", encoding="utf-8")
    (raw / "outside.txt").write_text("Autre document.", encoding="utf-8")
    processed = tmp_path / "processed"
    manifest = tmp_path / "ingestion.jsonl"
    records = {"guide.txt": {"source_url": "https://example.test/guide"}}

    first = ingest_directory(raw, processed, manifest, source_records=records)
    reused = ingest_directory(raw, processed, manifest, source_records=records)
    assert first["new_documents"] == 1
    assert reused["skipped_unchanged"] == 1

    records["guide.txt"]["source_url"] = "https://example.test/nouveau-guide"
    changed = ingest_directory(raw, processed, manifest, source_records=records)
    assert changed["new_documents"] == 1
    (output_file,) = processed.glob("*.json")
    output = json.loads(output_file.read_text(encoding="utf-8"))
    assert output["metadata"]["source_url"] == "https://example.test/nouveau-guide"
