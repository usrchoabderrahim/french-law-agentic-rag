import pytest

from src.retrieval.hybrid import rrf_fuse
from src.retrieval.loader import load_all
from src.retrieval.search import Retriever
from tests.helpers import FakeVectorStore, make_result


def test_rrf_prefers_chunks_found_by_both():
    vec = [make_result("a"), make_result("b"), make_result("c")]
    lex = [make_result("c"), make_result("a"), make_result("d")]
    assert [r.chunk_id for r in rrf_fuse([vec, lex], 4)] == ["a", "c", "b", "d"]


def _retriever():
    target = load_all()["fstbm"][22]
    return Retriever(FakeVectorStore(), lambda q: target.embedding.tolist()), target


def test_three_modes_return_the_target():
    r, target = _retriever()
    for mode in ("vector", "bm25", "hybrid"):
        res = r.search("inscription absent semestre", "fstbm", "reader", k=5, mode=mode)
        assert target.chunk_id in [x.chunk_id for x in res], mode
    assert r.search("inscription absent semestre", "fstbm", "reader", mode="hybrid")[0].method == "hybrid"


def test_hybrid_with_role_and_document_type():
    r, target = _retriever()
    res = r.search("inscription", "fstbm", "reader", k=5, mode="hybrid", document_type="normes_pedagogiques")
    assert res and all(x.document_type in (None, "normes_pedagogiques") for x in res)


def test_invalid_inputs():
    r, _ = _retriever()
    with pytest.raises(ValueError):
        r.search("  ", "fstbm", "reader")
    with pytest.raises(ValueError):
        r.search("x", "fstbm", "reader", mode="magic")
    with pytest.raises(PermissionError):
        r.search("x", "fstbm", None)

























# import pytest

# from src.retrieval.hybrid import rrf_fuse
# from src.retrieval.loader import load_all
# from src.retrieval.search import Retriever
# from tests.helpers import FakeVectorStore, make_result


# def test_rrf_prefers_chunks_found_by_both():
#     vec = [make_result("a"), make_result("b"), make_result("c")]
#     lex = [make_result("c"), make_result("a"), make_result("d")]
#     assert [r.chunk_id for r in rrf_fuse([vec, lex], 4)] == ["a", "c", "b", "d"]


# def _retriever():
#     target = load_all()["fstbm"][22]
#     return Retriever(FakeVectorStore(), lambda q: target.embedding.tolist()), target


# def test_three_modes_return_the_target():
#     r, target = _retriever()
#     for mode in ("vector", "bm25", "hybrid"):
#         res = r.search("inscription absent semestre", "fstbm", "lecteur", k=5, mode=mode)
#         assert target.chunk_id in [x.chunk_id for x in res], mode
#     assert r.search("inscription absent semestre", "fstbm", "lecteur", mode="hybrid")[0].method == "hybrid"


# def test_hybrid_with_role_and_document_type():
#     r, target = _retriever()
#     res = r.search("inscription", "fstbm", "user", k=5, mode="hybrid", document_type="normes_pedagogiques")
#     assert res and all(x.document_type in (None, "normes_pedagogiques") for x in res)


# def test_invalid_inputs():
#     r, _ = _retriever()
#     with pytest.raises(ValueError):
#         r.search("  ", "fstbm", "lecteur")
#     with pytest.raises(ValueError):
#         r.search("x", "fstbm", "lecteur", mode="magic")
#     with pytest.raises(PermissionError):
#         r.search("x", "fstbm", None)
