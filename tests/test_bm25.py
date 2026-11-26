from src.retrieval.bm25 import Bm25Store, tokenize
from src.retrieval.loader import load_all
from tests.helpers import make_chunk


def test_tokenize_removes_accents_and_stopwords():
    assert tokenize("L'inscription pédagogique des étudiants") == ["inscription", "pedagogique", "etudiants"]


def test_bm25_finds_real_article():
    hits = Bm25Store(load_all()).search("inscription pédagogique étudiant absent semestre", "fstbm", "reader", 3)
    assert "Article 5.1" in [h.section for h in hits]


def test_bm25_works_on_ocr_corpus():
    hits = Bm25Store(load_all()).search("dahir", "ocr", "reader", 3)
    assert len(hits) > 0 and all(h.corpus == "ocr" for h in hits)


def test_bm25_unknown_word_returns_nothing():
    assert Bm25Store(load_all()).search("zzzzqqqq", "fstbm", "reader", 3) == []




































# from src.retrieval.bm25 import Bm25Store, tokenize
# from src.retrieval.loader import load_all
# from tests.helpers import make_chunk


# def test_tokenize_removes_accents_and_stopwords():
#     assert tokenize("L'inscription pédagogique des étudiants") == ["inscription", "pedagogique", "etudiants"]


# def test_bm25_finds_real_article():
#     hits = Bm25Store(load_all()).search("inscription pédagogique étudiant absent semestre", "fstbm", "lecteur", 3)
#     assert "Article 5.1" in [h.section for h in hits]


# def test_bm25_works_on_ocr_corpus():
#     hits = Bm25Store(load_all()).search("dahir", "ocr", "lecteur", 3)
#     assert len(hits) > 0 and all(h.corpus == "ocr" for h in hits)


# def test_bm25_unknown_word_returns_nothing():
#     assert Bm25Store(load_all()).search("zzzzqqqq", "fstbm", "lecteur", 3) == []
