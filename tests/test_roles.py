import pytest

from src.retrieval.bm25 import Bm25Store
from src.retrieval.loader import load_all
from src.retrieval.vector_store import build_where, check_role
from tests.helpers import make_chunk


def test_role_is_mandatory():
    for bad in (None, "", "hacker", "lecteur", "user"):
        with pytest.raises(PermissionError):
            check_role(bad)
    with pytest.raises(PermissionError):
        build_where(None)
    with pytest.raises(PermissionError):
        Bm25Store(load_all()).search("budget", "fstbm", None, 3)


def test_only_roles_of_the_data_exist():
    assert check_role("admin") == "admin" and check_role("reader") == "reader"


def test_restricted_chunk_is_never_returned():
    chunks = [make_chunk("a", "secret budget", ["admin"]),
              make_chunk("b", "budget public", ["admin", "reader"])]
    store = Bm25Store({"fstbm": chunks})
    assert [h.chunk_id for h in store.search("budget", "fstbm", "reader", 5)] == ["b"]
    assert {h.chunk_id for h in store.search("budget", "fstbm", "admin", 5)} == {"a", "b"}


















































# import pytest

# from src.retrieval.bm25 import Bm25Store
# from src.retrieval.loader import load_all
# from src.retrieval.vector_store import build_where, check_role
# from tests.helpers import make_chunk


# def test_role_is_mandatory():
#     for bad in (None, "", "hacker"):
#         with pytest.raises(PermissionError):
#             check_role(bad)
#     with pytest.raises(PermissionError):
#         build_where(None)
#     with pytest.raises(PermissionError):
#         Bm25Store(load_all()).search("budget", "fstbm", None, 3)


# def test_user_is_alias_of_lecteur():
#     assert check_role("user") == "lecteur" and check_role("reader") == "lecteur"


# def test_restricted_chunk_is_never_returned():
#     chunks = [make_chunk("a", "secret budget", ["admin"]),
#               make_chunk("b", "budget public", ["admin", "lecteur"])]
#     store = Bm25Store({"fstbm": chunks})
#     assert [h.chunk_id for h in store.search("budget", "fstbm", "lecteur", 5)] == ["b"]
#     assert [h.chunk_id for h in store.search("budget", "fstbm", "user", 5)] == ["b"]
#     assert {h.chunk_id for h in store.search("budget", "fstbm", "admin", 5)} == {"a", "b"}
