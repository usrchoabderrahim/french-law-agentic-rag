"""outil search_corpus, chercheur, verificateur."""
import numpy as np
import pytest

from src.agent import config
from src.agent.graph import AgenticRAG
from src.agent.nodes.chercheur import article_filter, make_chercheur, new_queries
from src.agent.nodes.verificateur import code_check, decide, make_verificateur
from src.agent.state import new_state, new_sub_result
from src.agent.tools import NumpyVectorStore, SearchTool, ToolError, build_search_tool
from src.retrieval.loader import load_all
from src.retrieval.search import Retriever
from tests.agent.fakes import FakeSearchTool, ScriptedLLM, passage, requires_data
from tests.helpers import make_chunk

THESE = ("La durée de préparation de la thèse est de trois ans au minimum et de six ans "
         "au maximum, selon l'article 5 du décret du 12 mars 2008.")


def _state(sub_text="Quelle est la durée de la thèse ?"):
    state = new_state(sub_text, "reader")
    state["sub_results"] = {"q1": new_sub_result("q1", sub_text)}
    return state


def _search_then_verify(state, llm, tool):
    state.update(make_chercheur(llm, tool)(state))
    state.update(make_verificateur(llm)(state))
    return state["sub_results"]["q1"]


# ------------------------------------------------------------------ outil RAG
@pytest.fixture(scope="module")
def real_tool():
    corpora = load_all()
    return SearchTool(Retriever(vector_store=NumpyVectorStore(corpora), encode=lambda q: []),
                      mode="bm25")


@requires_data
def test_embeddings_charges_et_alignes():
    corpora = load_all()
    assert {k: len(v) for k, v in corpora.items()} == {"fstbm": 44, "ocr": 981}
    store = NumpyVectorStore(corpora)
    assert store.matrices["ocr"].shape == (981, 1024)
    norms = np.linalg.norm(store.matrices["fstbm"], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)                  # bge-m3 normalisé


@requires_data
def test_recherche_vectorielle_reelle_retrouve_le_chunk():
    corpora = load_all()
    store = NumpyVectorStore(corpora)
    for name in ("fstbm", "ocr"):
        chunk = corpora[name][7]
        top = store.search(chunk.embedding, name, "reader", 3)
        assert top[0].chunk_id == chunk.chunk_id and top[0].score == pytest.approx(1, abs=1e-3)


@requires_data
@pytest.mark.parametrize("query,corpus", [("inscription doctorat thèse", "fstbm"),
                                          ("bulletin officiel dahir société", "ocr")])
def test_search_corpus_deux_corpus(real_tool, query, corpus):
    hits = real_tool.call("search_corpus", {"query": query}, user_role="reader", corpora=[corpus])
    assert hits and all(h["corpus"] == corpus for h in hits)
    assert {"chunk_id", "text", "document", "page"} <= hits[0].keys()
    both = real_tool.call("search_corpus", {"query": query}, user_role="admin",
                          corpora=["fstbm", "ocr"])
    assert both and len(both) <= config.TOP_K


@requires_data
def test_outil_liste_blanche_et_role_impose_par_le_wrapper(real_tool):
    with pytest.raises(ToolError):
        real_tool.call("delete_corpus", {"query": "x"}, user_role="reader", corpora=["fstbm"])
    # le LLM tente d'imposer un rôle : l'argument est ignoré, le rôle vient de l'état
    with pytest.raises(PermissionError):
        real_tool.call("search_corpus", {"query": "thèse", "user_role": "admin"},
                       user_role=None, corpora=["fstbm"])


def test_document_restreint_jamais_retourne():
    corpora = {"fstbm": [make_chunk("secret", "budget confidentiel thèse", ["admin"]),
                         make_chunk("public", "budget public thèse", ["admin", "reader"])]}
    corpora["fstbm"][0].embedding = np.ones(1024, dtype=np.float32) / 32
    corpora["fstbm"][1].embedding = np.ones(1024, dtype=np.float32) / 32
    retriever = Retriever.__new__(Retriever)
    from src.retrieval.bm25 import Bm25Store
    retriever.corpora, retriever.bm25 = corpora, Bm25Store(corpora)
    retriever.vector_store, retriever.encode = NumpyVectorStore(corpora), lambda q: np.ones(1024) / 32
    for mode in ("bm25", "vector", "hybrid"):
        tool = SearchTool(retriever, mode=mode)
        reader = tool.call("search_corpus", {"query": "budget thèse"}, user_role="reader",
                           corpora=["fstbm"])
        admin = tool.call("search_corpus", {"query": "budget thèse"}, user_role="admin",
                          corpora=["fstbm"])
        assert [h["chunk_id"] for h in reader] == ["public"]
        assert {h["chunk_id"] for h in admin} == {"secret", "public"}


@requires_data
def test_build_search_tool_reel():
    tool = build_search_tool()
    assert tool.mode in ("bm25", "hybrid")
    assert tool.call("search_corpus", {"query": "charte de thèse"}, user_role="reader",
                     corpora=["fstbm"])


# ------------------------------------------------------------------ chercheur
def test_requetes_jamais_repetees():
    assert new_queries(["durée thèse", "DURÉE  thèse", "nouvelle"], "q", ["Durée thèse"],
                       first=False) == ["nouvelle"]
    assert new_queries(["a", "b", "c", "d"], "sous-question", [], first=True) == [
        "sous-question", "a", "b"]                                  # ≤ 3, sous-question en tête


@pytest.mark.parametrize("text,expected", [
    ("Que dit l'article 5 ?", {"article_number": "5"}),
    ("Que prévoit l'article premier ?", {"article_number": "PREMIER"}),
    ("Et l'article 2.3 des normes ?", {"article_number": "2.3"}),
    ("Quelle durée ?", None)])
def test_filtre_article_exact(text, expected):
    assert article_filter(text) == expected


def test_role_ajoute_par_le_wrapper_et_rejets_ecartes():
    tool = FakeSearchTool([passage("c1", THESE), passage("c2", "Texte hors sujet sur les bourses.")])
    state = _state()
    state["sub_results"]["q1"]["rejected_chunk_ids"] = ["c2"]
    llm = ScriptedLLM()
    state.update(make_chercheur(llm, tool)(state))
    sub = state["sub_results"]["q1"]
    assert all(c["user_role"] == "reader" for c in tool.calls)
    assert all("user_role" not in c["args"] for c in tool.calls)
    req = next(d for t, d in llm.calls if t == "chercheur_requetes")
    assert "user_role" not in req                                  # invisible pour le LLM
    assert [p["chunk_id"] for p in sub["passages"]] == ["c1"]
    assert sub["attempts"] == 1 and sub["cited_chunk_ids"] == ["c1"]


def test_limite_appels_outil_par_tentative(monkeypatch):
    monkeypatch.setattr(config, "MAX_TOOL_CALLS_PER_ATTEMPT", 2)
    tool = FakeSearchTool([])
    llm = ScriptedLLM({"chercheur_requetes": [{"queries": ["a b c", "d e f", "g h i"]}]})
    _search_then_verify(_state("Que dit l'article 5 ?"), llm, tool)
    assert len(tool.calls) == 2


def test_rien_trouve_reponse_vide_puis_insuffisant():
    llm = ScriptedLLM()
    state = _state()
    tool = FakeSearchTool([])
    for attempt in range(1, config.MAX_RESEARCH_ATTEMPTS + 1):
        sub = _search_then_verify(state, llm, tool)
        assert sub["partial_answer"] == "" and sub["attempts"] == attempt
    assert sub["status"] == "insuffisant"
    tried = [c["args"]["query"].lower() for c in tool.calls]
    assert len(tried) == len(set(tried))                            # aucune requête répétée


# ------------------------------------------------------------------ verificateur
def test_seuils_et_confiance():
    assert decide(0.8, 0.6) == (True, 0.7)
    assert decide(0.79, 0.9)[0] is False and decide(0.9, 0.59)[0] is False
    assert decide(1.0, 0.5)[1] == 0.75


def test_controle_par_code_chiffres_dates_articles():
    ok, missing = code_check("Trois ans selon l'article 5 du décret du 12 mars 2008.", THESE)
    assert ok == 1.0 and missing == []
    ratio, missing = code_check("Selon l'article 9, la durée est de 4 ans.", THESE)
    assert ratio < 1 and {"article 9", "4"} <= set(missing)


def test_verificateur_rejette_reponse_non_etayee_puis_retente():
    """Le LLM vérificateur est trop indulgent : le code détecte le chiffre inventé."""
    tool = FakeSearchTool([passage("c1", THESE)])
    llm = ScriptedLLM({
        "chercheur_reponse": [{"partial_answer": "La thèse dure 10 ans (article 9).",
                               "cited_chunk_ids": ["c1"]},
                              {"partial_answer": THESE, "cited_chunk_ids": ["c1"]}],
        "verificateur": [{"supported_ratio": 1.0, "relevance": 0.9, "feedback": "ok"}],
    })
    state = _state()
    sub = _search_then_verify(state, llm, tool)
    assert sub["verdict"]["valid"] is False and sub["status"] == "en_cours"
    assert sub["verdict"]["supported_ratio"] == 0.0
    assert any("10" in c for c in sub["verdict"]["unsupported_claims"])
    sub = _search_then_verify(state, llm, tool)                     # nouvelle tentative
    assert sub["status"] == "valide" and sub["attempts"] == 2
    assert sub["verdict"]["confidence"] == pytest.approx(0.95)
    second = [d for t, d in llm.calls if t == "chercheur_requetes"][1]
    assert second["queries_tried"] and second["feedback"] == "ok"   # lit le verdict


def test_verificateur_pertinence_insuffisante():
    tool = FakeSearchTool([passage("c1", THESE)])
    llm = ScriptedLLM({"verificateur": [{"supported_ratio": 1.0, "relevance": 0.3,
                                         "missing_aspects": ["prolongation"]}]})
    state = _state()
    for _ in range(config.MAX_RESEARCH_ATTEMPTS):
        sub = _search_then_verify(state, llm, tool)
    assert sub["status"] == "insuffisant" and sub["attempts"] == config.MAX_RESEARCH_ATTEMPTS


def test_verificateur_json_invalide_rejet_prudent():
    tool = FakeSearchTool([passage("c1", THESE)])
    llm = ScriptedLLM({"verificateur": ["oops"]})
    sub = _search_then_verify(_state(), llm, tool)
    assert sub["verdict"]["valid"] is False and sub["verdict"]["confidence"] == 0.0


@requires_data
def test_sous_graphe_recherche_sur_corpus_reel(real_tool):
    out = AgenticRAG(llm=ScriptedLLM(), search_tool=real_tool).run(
        "Quelles sont les conditions d'inscription en thèse de doctorat ?", "reader",
        corpora=["fstbm"])
    sub = out["sub_results"]["q1"]
    assert sub["passages"] and sub["status"] in ("valide", "insuffisant")
