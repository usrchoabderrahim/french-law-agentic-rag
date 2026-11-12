from ingestion.hashing import sha256_file, stable_document_id


def test_sha256_and_stable_id(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("texte juridique", encoding="utf-8")
    digest = sha256_file(path)
    assert len(digest) == 64
    assert stable_document_id(digest) == stable_document_id(digest)
