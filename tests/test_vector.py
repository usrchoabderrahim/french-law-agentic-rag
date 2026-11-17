from src.retrieval.loader import load_all
from src.retrieval.vector_store import build_where, to_metadata


def test_where_clause_role_only():
    assert build_where("reader") == {"role_reader": True}


def test_where_clause_role_and_document_type():
    assert build_where("reader", {"document_type": "charte"}) == {
        "$and": [{"role_reader": True}, {"document_type": "charte"}]
    }


def test_metadata_is_chroma_compatible():
    meta = to_metadata(load_all()["fstbm"][0])
    assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())
    assert meta["role_reader"] is True and meta["role_admin"] is True














# from src.retrieval.loader import load_all
# from src.retrieval.vector_store import build_where, to_metadata


# def test_where_clause_role_only():
#     assert build_where("lecteur") == {"role_lecteur": True}


# def test_where_clause_role_and_document_type():
#     assert build_where("lecteur", {"document_type": "charte"}) == {
#         "$and": [{"role_lecteur": True}, {"document_type": "charte"}]
#     }


# def test_metadata_is_chroma_compatible():
#     meta = to_metadata(load_all()["fstbm"][0])
#     assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())
#     assert meta["role_lecteur"] is True and meta["role_editeur"] is False
