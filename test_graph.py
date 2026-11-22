import pytest

from src.retrieval.graph_rag import ARTICLE, ORG, SIGLE, TEXTE, GraphStore, LegalGraph, RuleExtractor
from src.retrieval.loader import load_all
from tests.helpers import make_chunk


def keys(text, typ=None, known=None):
    mentions, _ = RuleExtractor(known).extract(text)
    return {m.key for m in mentions if typ is None or m.type == typ}


def test_extract_legal_texts():
    text = "Vu le dahir n° 1-58-376 du 15 novembre 1958 et la loi n° 16-98 ; décret n° 2-01-1643."
    assert {"dahir 1-58-376", "loi 16-98", "decret 2-01-1643"} <= keys(text, TEXTE)


def test_extract_dated_text_and_arrete_with_number():
    assert "arrete viziriel 10 aout 1955" in keys("Arrêté viziriel du 10 août 1955 autorisant", TEXTE)
    assert "arrete 1318-04" in keys("Arrêté du ministre de la santé n° 1318-04 du 11 joumada", TEXTE)


def test_extract_articles_incl_decimal_and_premier():
    assert keys("Article 2.3 et ARTICLE PREMIER, art. 59", ARTICLE) == {"article 2.3", "article 1", "article 59"}


def test_extract_organisations_upper_and_lower_case():
    assert "direction de la sante publique et de la famille" in keys(
        "DIRECTION DE LA SANTÉ PUBLIQUE ET DE LA FAMILLE.", ORG)
    org = keys("arrêté du ministre de l'énergie et des mines est chargé", ORG)
    assert "ministre de l energie et des mines" in org


def test_sigle_filters_common_words_and_roman_numerals():
    text = "Le CNPN et la DIRECTION. Section III."
    assert keys(text, SIGLE, known={"direction"}) == {"cnpn"}


def test_typed_relations_active_and_passive():
    _, rel = RuleExtractor().extract("Dahir du 13 septembre 1955 modifiant le dahir du 23 janvier 1937.")
    assert [(r.source, r.relation, r.target) for r in rel] == [
        ("dahir 13 septembre 1955", "modifie", "dahir 23 janvier 1937")]
    _, rel = RuleExtractor().extract("le dahir du 23 janvier 1937 modifié par le dahir du 13 septembre 1955")
    assert [(r.source, r.relation, r.target) for r in rel] == [
        ("dahir 13 septembre 1955", "modifie", "dahir 23 janvier 1937")]


def _chunks():
    return [
        make_chunk("a", "La loi n° 16-98 fixe les conditions du registre.", ["admin", "reader"]),
        make_chunk("b", "Le registre est tenu par le tribunal. Loi n° 16-98 article 10.", ["admin", "reader"]),
        make_chunk("c", "Texte sans rapport sur la comptabilité.", ["admin", "reader"]),
        make_chunk("d", "Dahir n° 1-58-376 confidentiel et loi n° 16-98.", ["admin"]),
    ]


def test_graph_search_finds_chunks_citing_the_entity():
    graph = LegalGraph.build(_chunks()[:3])
    hits = [c.chunk_id for c, _ in graph.search("Que dit la loi n° 16-98 ?", 5)]
    assert set(hits) == {"a", "b"}


def test_graph_returns_nothing_without_known_entity():
    assert LegalGraph.build(_chunks()[:3]).search("question vague sans entité", 5) == []


def test_graph_store_applies_role_before_building():
    store = GraphStore({"fstbm": _chunks()})
    reader = {r.chunk_id for r in store.search("loi n° 16-98", "fstbm", "reader", 10)}
    admin = {r.chunk_id for r in store.search("loi n° 16-98", "fstbm", "admin", 10)}
    assert "d" not in reader and "d" in admin
    # aucune entité du chunk interdit ne fuit dans le graphe du lecteur
    assert "dahir 1-58-376" not in store.graph("fstbm", "reader").types


def test_graph_store_requires_valid_role_and_corpus():
    store = GraphStore({"fstbm": _chunks()})
    for bad in (None, "", "hacker"):
        with pytest.raises(PermissionError):
            store.search("loi n° 16-98", "fstbm", bad, 3)
    with pytest.raises(ValueError):
        store.search("x", "inconnu", "reader", 3)


def test_graph_store_applies_metadata_filters():
    chunks = [make_chunk("a", "loi n° 16-98", ["reader"], doc_type="guide"),
              make_chunk("b", "loi n° 16-98", ["reader"], doc_type="charte")]
    res = GraphStore({"fstbm": chunks}).search("loi n° 16-98", "fstbm", "reader", 5, {"document_type": "charte"})
    assert [r.chunk_id for r in res] == ["b"]


def test_graph_on_real_corpora_builds_and_finds_article():
    corpora = load_all()
    store = GraphStore(corpora)
    for name in corpora:
        stats = store.graph(name, "reader").stats()
        assert stats["entities"] > 0 and stats["chunks"] == len(corpora[name])
    res = store.search("délai article 2.3 fraude", "fstbm", "reader", 3)
    assert res and "Article 2.3" in res[0].text
