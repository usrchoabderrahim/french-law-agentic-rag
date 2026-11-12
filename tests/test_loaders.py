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


def test_import_html_prefers_main_content(tmp_path):
    path = tmp_path / "inscription.html"
    path.write_text(
        """
        <html>
        <head>
            <title>Guide E-Scolarité</title>
            <meta name="date" content="2024-03-12">
        </head>
        <body>
            <header>Nom du site</header>
            <main>
                <div id="wrap">
                    <h1>Inscription</h1>
                    <p>Créez votre compte.</p>
                    <nav>Navigation du guide</nav>
                </div>
            </main>
            <footer>Adresse du site</footer>
        </body>
        </html>
        """,
        encoding="utf-8",
    )

    result = load_document(path)

    assert result.pages[0].text == "Inscription\nCréez votre compte."
    assert result.metadata.title == "Guide E-Scolarité"
    assert result.metadata.publication_date == date(2024, 3, 12)


def _insert_scan_image(page, rect=None):
    image = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 64, 64), False)
    image.clear_with(200)
    page.insert_image(rect or page.rect, pixmap=image, keep_proportion=False)


def test_import_scanned_pdf_uses_ocr(tmp_path, monkeypatch):
    path = tmp_path / "scan.pdf"

    pdf = pymupdf.open()
    _insert_scan_image(pdf.new_page())
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


def test_import_mixed_pdf_routes_pages_and_preserves_native_fallback(tmp_path, monkeypatch):
    path = tmp_path / "mixed.pdf"
    native_text = "Guide court."
    hidden_header = "FST Béni Mellal. " * 10
    body_text = "Le texte natif du guide reste disponible. " * 3

    with pymupdf.open() as pdf:
        short_native = pdf.new_page()
        short_native.insert_text((30, 45), native_text)
        _insert_scan_image(short_native, pymupdf.Rect(10, 10, 20, 20))

        incomplete_layer = pdf.new_page()
        _insert_scan_image(incomplete_layer)
        incomplete_layer.insert_textbox(pymupdf.Rect(30, 20, 560, 70), hidden_header, render_mode=3)

        complete_layer = pdf.new_page()
        _insert_scan_image(complete_layer)
        complete_layer.insert_textbox(pymupdf.Rect(30, 300, 560, 400), body_text)

        pdf.new_page()

        failed = pdf.new_page()
        _insert_scan_image(failed)
        failed.insert_text((30, 45), native_text)

        _insert_scan_image(pdf.new_page())

        final_native = pdf.new_page()
        final_native.insert_text((30, 45), "Dernière page.")
        pdf.save(path)

    attempted = []

    def recognize(page):
        attempted.append(page.number + 1)
        if page.number == 4:
            raise RuntimeError("Tesseract process timeout")
        return "" if page.number == 5 else "Texte complet du document scanné."

    monkeypatch.setattr("ingestion.loaders._ocr_pdf_page", recognize)

    result = load_document(path)

    assert attempted == [2, 5, 6]
    assert [page.page_number for page in result.pages] == list(range(1, 8))
    assert result.pages[0].text.strip() == native_text
    assert result.pages[1].text == "Texte complet du document scanné."
    assert "Le texte natif du guide" in result.pages[2].text
    assert not result.pages[3].text.strip()
    assert result.pages[4].text.strip() == native_text
    assert not result.pages[5].text.strip()
    assert result.pages[6].text.strip() == "Dernière page."

    ocr = result.metadata.extra["ocr"]
    assert ocr["pages_attempted"] == [2, 5, 6]
    assert ocr["pages_with_text"] == [2]
    assert ocr["pages_failed"] == [5, 6]
    assert ocr["pages_failed_count"] == 2
    diagnostics = ocr["page_diagnostics"]
    expected_sources = ["native", "ocr", "native", "blank", "failed", "failed", "native"]
    assert [page["source"] for page in diagnostics] == expected_sources
    assert diagnostics[1]["reason"] == "incomplete_native_text_layer"
    assert diagnostics[1]["duration_seconds"] >= 0
    assert diagnostics[4]["error"] == "Tesseract process timeout"


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
