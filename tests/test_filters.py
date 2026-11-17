from src.retrieval.bm25 import Bm25Store
from src.retrieval.loader import load_all
from src.retrieval.search import Retriever
from tests.helpers import FakeVectorStore, make_chunk


def test_document_type_filter_bm25():
    chunks = [make_chunk("a", "inscription etudiant", ["admin", "reader"], "charte"),
              make_chunk("b", "inscription etudiant", ["admin", "reader"], "guide")]
    hits = Bm25Store({"fstbm": chunks}).search("inscription", "fstbm", "reader", 5, {"document_type": "guide"})
    assert [h.chunk_id for h in hits] == ["b"]


def test_document_type_filter_real_data():
    store = Bm25Store(load_all())
    hits = store.search("inscription", "fstbm", "reader", 10, {"document_type": "guide"})
    assert hits and all(h.document_type == "guide" for h in hits)


def test_document_types_listed_for_front():
    r = Retriever(FakeVectorStore(), lambda q: [0.0] * 1024)
    assert "charte" in r.document_types("fstbm") and r.document_types("ocr") == ["bulletin_officiel"]


























# from src.retrieval.bm25 import Bm25Store
# from src.retrieval.loader import load_all
# from src.retrieval.search import Retriever
# from tests.helpers import FakeVectorStore, make_chunk


# def test_document_type_filter_bm25():
#     chunks = [make_chunk("a", "inscription etudiant", ["admin", "lecteur"], "charte"),
#               make_chunk("b", "inscription etudiant", ["admin", "lecteur"], "guide")]
#     hits = Bm25Store({"fstbm": chunks}).search("inscription", "fstbm", "lecteur", 5, {"document_type": "guide"})
#     assert [h.chunk_id for h in hits] == ["b"]


# def test_document_type_filter_real_data():
#     store = Bm25Store(load_all())
#     hits = store.search("inscription", "fstbm", "lecteur", 10, {"document_type": "guide"})
#     assert hits and all(h.document_type == "guide" for h in hits)


# def test_document_types_listed_for_front():
#     r = Retriever(FakeVectorStore(), lambda q: [0.0] * 1024)
#     assert "charte" in r.document_types("fstbm") and r.document_types("ocr") == ["bulletin_officiel"]
