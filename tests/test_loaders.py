from datetime import date

import pymupdf
from docx import Document

from ingestion.loaders import load_document


def test_import_pdf(tmp_path):
    path = tmp_path / "test.pdf"

    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Texte juridique PDF")
    pdf.save(path)
    pdf.close()

    result = load_document(path)

    assert "Texte juridique PDF" in result.pages[0].text
    assert result.metadata.file_type == "pdf"
    assert result.metadata.page_count == 1


def test_import_docx(tmp_path):
    path = tmp_path / "test.docx"

    doc = Document()
    doc.add_paragraph("Texte juridique DOCX")
    doc.core_properties.title = "Loi marocaine"
    doc.save(path)

    result = load_document(path)

    assert "Texte juridique DOCX" in result.pages[0].text
    assert result.metadata.title == "Loi marocaine"
    assert result.metadata.file_type == "docx"


def test_import_txt(tmp_path):
    path = tmp_path / "test.txt"
    path.write_text("Texte juridique TXT", encoding="utf-8")

    result = load_document(path)

    assert "Texte juridique TXT" in result.pages[0].text
    assert result.metadata.file_type == "txt"


def test_import_markdown(tmp_path):
    path = tmp_path / "test.md"
    path.write_text("# Loi marocaine\nArticle 1", encoding="utf-8")

    result = load_document(path)

    assert "Article 1" in result.pages[0].text
    assert result.metadata.file_type == "md"


def test_import_html(tmp_path):
    path = tmp_path / "test.html"

    path.write_text(
        """
        <html>
        <head>
            <title>Loi marocaine</title>
        </head>
        <body>
            <p>Article 1 : texte juridique</p>
        </body>
        </html>
        """,
        encoding="utf-8",
    )

    result = load_document(path)

    assert result.metadata.title == "Loi marocaine"
    assert "Article 1" in result.pages[0].text
    assert result.metadata.file_type == "html"


def test_import_html_extracts_publication_date(tmp_path):
    path = tmp_path / "loi.html"

    path.write_text(
        """
        <html>
        <head>
            <title>Loi marocaine</title>
            <meta property="article:published_time"
                  content="2024-03-12T09:30:00+01:00">
        </head>
        <body>
            <p>Article 1 : texte juridique</p>
        </body>
        </html>
        """,
        encoding="utf-8",
    )

    result = load_document(
        path,
        source_name="SGG",
        source_url="https://www.sgg.gov.ma/",
    )

    assert result.metadata.publication_date == date(2024, 3, 12)
    assert result.metadata.publication_date_raw == "2024-03-12T09:30:00+01:00"
    assert result.metadata.source_name == "SGG"
    assert result.metadata.source_url == "https://www.sgg.gov.ma/"


def test_import_scanned_pdf_uses_ocr(tmp_path, monkeypatch):
    path = tmp_path / "scan.pdf"

    pdf = pymupdf.open()
    pdf.new_page()
    pdf.save(path)
    pdf.close()

    monkeypatch.setattr(
        "ingestion.loaders._ocr_pdf_page",
        lambda page: "Texte juridique scanné",
    )

    result = load_document(path)

    assert result.pages[0].text == "Texte juridique scanné"

    assert result.metadata.extra["ocr"]["pages_attempted_count"] == 1

    assert result.metadata.extra["ocr"]["pages_with_text_count"] == 1

    assert result.metadata.extra["ocr"]["pages_with_text"] == [1]


def test_import_ocr_txt_preserves_page_markers(tmp_path):
    path = tmp_path / "ocr_document.txt"

    path.write_text(
        "PAGE 1\n" "Première page juridique.\n" "PAGE 2\n" "Deuxième page juridique.\n",
        encoding="utf-8",
    )

    result = load_document(path)

    assert len(result.pages) == 2

    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2

    assert "Première page" in result.pages[0].text
    assert "Deuxième page" in result.pages[1].text

    assert result.metadata.page_count == 2

    assert result.metadata.extra["ocr"]["page_markers_detected"] is True
