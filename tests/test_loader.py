from src.retrieval.loader import load_all, normalize_roles


def test_corpora_are_separate():
    corpora = load_all()
    assert set(corpora) == {"fstbm", "ocr"}
    assert len(corpora["fstbm"]) == 44
    assert len(corpora["ocr"]) == 981
    assert all(c.embedding.shape == (1024,) for c in corpora["ocr"])


def test_roles_are_kept_as_in_the_data():
    assert normalize_roles(["reader", "admin"]) == ["admin", "reader"]
    for chunks in load_all().values():
        assert all(c.allowed_roles == ["admin", "reader"] for c in chunks)





















# from src.retrieval.loader import load_all, normalize_roles


# def test_corpora_are_separate():
#     corpora = load_all()
#     assert set(corpora) == {"fstbm", "ocr"}
#     assert len(corpora["fstbm"]) == 44
#     assert len(corpora["ocr"]) == 981
#     assert all(c.embedding.shape == (1024,) for c in corpora["ocr"])


# def test_roles_are_normalized():
#     assert normalize_roles(["reader", "admin"]) == ["admin", "lecteur"]
