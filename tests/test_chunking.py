from ingestion.chunking import chunk_document
from ingestion.schemas import DocumentMetadata, LoadedDocument, PageText


def make_document(text):
    metadata = DocumentMetadata(
        document_id="doc-test",
        title="Loi test",
        source_name="test",
        local_path="test.txt",
        file_type="txt",
        sha256="a" * 64,
    )
    return LoadedDocument(metadata=metadata, pages=[PageText(page_number=1, text=text)])


def test_fixed_chunking_has_overlap():
    doc = make_document("A" * 500)
    chunks = chunk_document(doc, strategy="fixed", chunk_size=200, overlap=40)
    assert len(chunks) >= 3
    assert chunks[0].text[-40:] == chunks[1].text[:40]


def test_section_chunking_preserves_article_heading():
    doc = make_document("Préambule\\nArticle 1\\nTexte un.\\nArticle 2\\nTexte deux.")
    chunks = chunk_document(doc, strategy="section", chunk_size=500, overlap=50)
    assert any("Article 1" in chunk.text for chunk in chunks)


def test_section_chunking_tracks_article_across_pages():
    from ingestion.schemas import PageText

    metadata = DocumentMetadata(
        document_id="multi-page-test",
        title="Loi multi-pages",
        source_name="test",
        local_path="test.pdf",
        file_type="pdf",
        sha256="b" * 64,
    )

    doc = LoadedDocument(
        metadata=metadata,
        pages=[
            PageText(
                page_number=1,
                text="Article 1\n" + ("A " * 350),
            ),
            PageText(
                page_number=2,
                text="Suite de l'article 1\n" + ("B " * 350) + "\nArticle 2\nTexte final.",
            ),
        ],
    )

    chunks = chunk_document(
        doc,
        strategy="section",
        chunk_size=500,
        overlap=80,
    )

    article_1_chunks = [chunk for chunk in chunks if chunk.metadata.get("article_number") == "1"]

    assert len(article_1_chunks) >= 2

    assert any(chunk.page_start == 1 and chunk.page_end == 2 for chunk in article_1_chunks)

    assert all(chunk.section == "Article 1" for chunk in article_1_chunks)


def test_section_chunking_extracts_article_number():
    doc = make_document(
        "Article 1er : Dispositions générales\n"
        + "Texte juridique.\n"
        + "Article 2 : Application\n"
        + "Autre texte juridique."
    )

    chunks = chunk_document(doc, strategy="section")

    assert any(chunk.metadata.get("article_number") == "1ER" for chunk in chunks)

    assert any(chunk.metadata.get("article_number") == "2" for chunk in chunks)


def test_chunk_metadata_preserves_source_pages():
    from ingestion.schemas import PageText

    metadata = DocumentMetadata(
        document_id="page-test",
        title="Test",
        source_name="SGG",
        local_path="test.pdf",
        file_type="pdf",
        sha256="c" * 64,
    )

    doc = LoadedDocument(
        metadata=metadata,
        pages=[
            PageText(page_number=1, text="Premier texte juridique."),
            PageText(page_number=2, text="Deuxième texte juridique."),
        ],
    )

    chunks = chunk_document(doc, strategy="section")

    assert chunks
    assert chunks[0].metadata["source_pages"] == [1, 2]
