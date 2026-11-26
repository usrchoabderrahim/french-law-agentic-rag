import json

import pytest

from src.retrieval.advanced_search import (
    AdvancedOptions,
    AdvancedRetriever,
    AdvancedSearchPipeline,
    weighted_rrf,
)
from src.retrieval.llm import LLMUnavailable
from src.retrieval.loader import load_all
from src.retrieval.search import Retriever
from tests.helpers import FakeVectorStore, make_result


class FakeLLM:
    def __init__(self, queries=(), hyde="", fail=False):
        self.queries, self.hyde, self.fail, self.calls = list(queries), hyde, fail, 0

    def generate(self, prompt, system=None, json_mode=False):
        self.calls += 1
        if self.fail:
            raise LLMUnavailable("down")
        return json.dumps({"queries": self.queries}) if json_mode else self.hyde


class FakeReranker:
    def __init__(self):
        self.queries = []

    def rerank(self, query, results, k=5):
        self.queries.append(query)
        return list(reversed(results))[:k]


TARGET_ROW = 22  # "Article 5.1 : inscription pédagogique ... absent au semestre précédent"


def _setup(llm, encoder_row=0):
    corpora = load_all()
    target = corpora["fstbm"][TARGET_ROW]
    vector_vec = corpora["fstbm"][encoder_row].embedding.tolist()
    retriever = Retriever(FakeVectorStore(), lambda text: vector_vec)
    return AdvancedRetriever(retriever, llm=llm), target


def test_weighted_rrf_weights_matter():
    a = [make_result("x"), make_result("y")]
    b = [make_result("y"), make_result("x")]
    assert [r.chunk_id for r in weighted_rrf([(1.0, a), (0.2, b)], 2)] == ["x", "y"]
    assert [r.chunk_id for r in weighted_rrf([(0.2, a), (1.0, b)], 2)] == ["y", "x"]
    assert weighted_rrf([(1.0, a)], 1)[0].method == "advanced"


def test_multi_query_rescues_a_question_with_no_lexical_match():
    llm = FakeLLM(queries=["inscription pédagogique étudiant absent semestre précédent"])
    adv, target = _setup(llm)
    only_bm25 = AdvancedOptions(multi_query=False, hyde=False, graph=False, base_mode="bm25")
    with_multi = AdvancedOptions(multi_query=True, hyde=False, graph=False, base_mode="bm25")
    question = "zzzz qqqq"  # aucun mot en commun avec le corpus
    assert adv.search(question, "fstbm", "reader", 5, only_bm25) == []
    ids = [r.chunk_id for r in adv.search(question, "fstbm", "reader", 5, with_multi)]
    assert target.chunk_id in ids


def test_hyde_adds_a_vector_ranking():
    llm = FakeLLM(hyde="Texte hypothétique sur l'inscription")
    adv, target = _setup(llm, encoder_row=TARGET_ROW)  # l'encodeur renvoie le vecteur de la cible
    opts = AdvancedOptions(multi_query=False, hyde=True, graph=False, base_mode="bm25")
    results, trace = adv.search_with_trace("zzzz qqqq", "fstbm", "reader", 5, opts)
    assert trace.hyde_text and trace.ranking_sizes["hyde"] > 0
    assert results[0].chunk_id == target.chunk_id


def test_graph_signal_is_used():
    adv, _ = _setup(FakeLLM())
    opts = AdvancedOptions(multi_query=False, hyde=False, graph=True, base_mode="bm25")
    results, trace = adv.search_with_trace("zzzz CNPN", "fstbm", "reader", 5, opts)
    assert trace.graph_seeds >= 1 and trace.ranking_sizes["graph"] > 0
    assert results and all(r.method == "advanced" for r in results)


def test_llm_down_falls_back_to_base_search():
    adv, _ = _setup(FakeLLM(fail=True))
    results, trace = adv.search_with_trace("inscription absent semestre", "fstbm", "reader", 5)
    assert results and trace.reformulations == [] and trace.hyde_text is None


def test_invalid_role_is_rejected_before_any_llm_call():
    llm = FakeLLM(queries=["x"])
    adv, _ = _setup(llm)
    for bad in (None, "hacker"):
        with pytest.raises(PermissionError):
            adv.search("inscription", "fstbm", bad)
    with pytest.raises(ValueError):
        adv.search("   ", "fstbm", "reader")
    assert llm.calls == 0


def test_filters_reach_every_ranking():
    adv, _ = _setup(FakeLLM(queries=["inscription semestre"], hyde="inscription"))
    res = adv.search("inscription", "fstbm", "reader", 10, document_type="charte")
    types = {c.chunk_id: c.document_type for c in load_all()["fstbm"]}  # le faux store ne renvoie pas le type
    assert res and all(types[r.chunk_id] == "charte" for r in res)


def test_pipeline_reranks_with_the_original_question():
    adv, _ = _setup(FakeLLM(queries=["reformulation"]))
    reranker = FakeReranker()
    pipeline = AdvancedSearchPipeline(adv, reranker=reranker)
    results = pipeline.search("question originale inscription", "fstbm", "reader", k=3)
    assert reranker.queries == ["question originale inscription"] and len(results) == 3
    no_rerank = pipeline.search("inscription", "fstbm", "reader", k=3, options=AdvancedOptions(rerank=False))
    assert len(no_rerank) == 3 and reranker.queries == ["question originale inscription"]


def test_options_label():
    assert AdvancedOptions().label() == "hybrid+multi+hyde+graph+rerank"
    assert AdvancedOptions(False, False, False, False).label() == "hybrid"
