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


def test_section_chunking_preserves_dotted_article_numbers():
    doc = make_document(
        "Loi portant règlement des examens\n"
        "Article 1.1\nLes modalités sont communiquées aux étudiants.\n"
        "Article 1. 7 :\nLe rattrapage est organisé à la fin du semestre.\n"
        "Article 2,1 :\nL'assiduité est obligatoire.\n"
        "Article 4, 2 copies sont remises à chaque étudiant."
    )

    chunks = chunk_document(doc, strategy="section")
    articles = [chunk for chunk in chunks if chunk.metadata["heading_type"] == "ARTICLE"]

    assert [chunk.metadata["article_number"] for chunk in articles] == ["1.1", "1.7", "2.1", "4"]
    assert [chunk.metadata["article_id"].rsplit(":", 1)[-1] for chunk in articles] == [
        "1.1",
        "1.7",
        "2.1",
        "4",
    ]


def test_section_chunking_separates_roman_sections_and_preserves_pages():
    doc = make_document("")
    doc.pages = [
        PageText(
            page_number=2,
            text=(
                "Table des matières\n"
                "IV. Consultation des copies des épreuves .......... 4\n"
                "VI. Évaluation des activités para-universitaires .......... 5"
            ),
        ),
        PageText(
            page_number=5,
            text="Article 3.3\nLes sanctions varient selon les faits.\n1. Avertissement",
        ),
        PageText(
            page_number=6,
            text=(
                "La sanction est prononcée par le conseil.\n"
                "IV. Consultation des copies des épreuves\n"
                "La demande est déposée dans un délai maximal de 24 h.\n"
                "VI. Évaluation des activités para-universitaires\n"
                "Le jury attribue jusqu'à trois crédits.\n"
                "VII. Dispositions finales\n"
                "Les dispositions entrent en vigueur."
            ),
        ),
    ]

    chunks = chunk_document(doc, strategy="section")
    assert [chunk.metadata["heading_type"] for chunk in chunks] == [
        "TABLE_OF_CONTENTS",
        "ARTICLE",
        "SECTION",
        "SECTION",
        "SECTION",
    ]
    assert chunks[1].metadata["source_pages"] == [5, 6]
    assert "1. Avertissement" in chunks[1].text
    assert [chunk.section for chunk in chunks[2:]] == [
        "IV. Consultation des copies des épreuves",
        "VI. Évaluation des activités para-universitaires",
        "VII. Dispositions finales",
    ]
    assert all(chunk.metadata["article_number"] is None for chunk in chunks[2:])
    assert all(chunk.metadata["source_pages"] == [6] for chunk in chunks[2:])


def test_section_chunking_does_not_treat_body_prose_as_administrative_heading():
    doc = make_document(
        "Arrêté portant statut du cadre des inspecteurs\n"
        "Article 4\nLes candidats s'engagent à rester au\n"
        "service de l’État pendant une durée de huit ans.\n"
        "Article 6\nLes inspecteurs sont titularisés.\n"
        "DIRECTION DES AFFAIRES CHÉRIFIENNES\n"
        "Liste des nominations."
    )

    chunks = chunk_document(doc, strategy="section")

    assert [chunk.metadata["heading_type"] for chunk in chunks] == [
        "ACT",
        "ARTICLE",
        "ARTICLE",
        "DOCUMENT_SECTION",
    ]
    assert "service de l’État" in chunks[1].text
    assert chunks[2].metadata["article_id"] == "doc-test:act:0001:article:6"
    assert chunks[3].metadata["act_id"] is None


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
