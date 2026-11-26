"""Contribution 3 (Mouaad) : generateur_final, controle_citations, scénarios de bout en bout."""
import pytest

from src.agent import config
from src.agent.graph import AgenticRAG
from src.agent.nodes.controle_citations import check_citations
from src.agent.nodes.generateur_final import make_generateur_final, number_passages
from src.agent.state import new_state, new_sub_result
from tests.agent.fakes import FakeSearchTool, ScriptedLLM, passage, requires_data

P1 = "La durée de préparation de la thèse est de trois ans au minimum."
P2 = "Le doctorant présente chaque année un rapport d'avancement au centre des études doctorales."


def _subs():
    q1, q2, q3 = (new_sub_result(f"q{i}", t) for i, t in
                  enumerate(["Durée de la thèse ?", "Obligations du doctorant ?",
                             "Montant de la bourse ?"], start=1))
    q1.update(status="valide", passages=[passage("c1", P1, 3), passage("c9", "autre")],
              cited_chunk_ids=["c1"], partial_answer=P1, verdict={"confidence": 0.9})
    q2.update(status="valide", passages=[passage("c2", P2, 4), passage("c1", P1, 3)],
              cited_chunk_ids=["c2", "c1"], partial_answer=P2, verdict={"confidence": 0.8})
    q3.update(status="insuffisant", verdict={"confidence": 0.1})
    return {"q1": q1, "q2": q2, "q3": q3}


GOOD = {"answer": "Trois ans minimum [1]. Rapport annuel [2].",
        "citations": [{"n": 1, "chunk_id": "c1", "document": "Charte", "page": 3,
                       "extract": "trois ans au minimum"},
                      {"n": 2, "chunk_id": "c2", "document": "Charte", "page": 4,
                       "extract": "rapport d'avancement"}]}


# ------------------------------------------------------------------ generateur_final
def test_numerotation_par_le_code_sans_doublon():
    numbered = number_passages(_subs())
    assert [(p["n"], p["chunk_id"]) for p in numbered] == [(1, "c1"), (2, "c2")]


def test_generateur_entree_et_section_non_trouvee():
    llm = ScriptedLLM({"generateur_final": [GOOD]})
    state = new_state("q", "reader")
    state["sub_results"] = _subs()
    state.update(make_generateur_final(llm)(state))
    data = llm.calls[0][1]
    assert [p["n"] for p in data["passages"]] == [1, 2]
    assert data["insufficient"] == ["Montant de la bourse ?"]
    assert state["generation_attempts"] == 1
    assert config.NOT_FOUND_PREFIX + "Montant de la bourse ?" in state["final_answer"]


def test_heuristique_generateur_passe_le_controle():
    state = new_state("q", "reader")
    state["sub_results"] = _subs()
    state.update(make_generateur_final(ScriptedLLM())(state))
    assert check_citations(state["final_answer"], state["citations"], state["sub_results"]) == []
    assert "[1]" in state["final_answer"] and "[2]" in state["final_answer"]


# ------------------------------------------------------------------ controle_citations
def test_citations_valides():
    assert check_citations(GOOD["answer"], GOOD["citations"], _subs()) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda a, c: (a + " Voir aussi [7].", c), "[7] utilisé"),                       # n inconnu
    (lambda a, c: (a, [{**c[0], "chunk_id": "c_invente"}, c[1]]), "inventé"),        # chunk inventé
    (lambda a, c: (a, [{**c[0], "page": None}, c[1]]), "page"),                      # page manquante
    (lambda a, c: (a, [{**c[0], "extract": ""}, c[1]]), "extract"),                  # extrait manquant
    (lambda a, c: (a, [{**c[0], "extract": "cinq ans au maximum"}, c[1]]), "ne figure pas"),
    (lambda a, c: ("", c), "réponse vide"),
])
def test_citations_fabriquees_detectees(mutate, needle):
    answer, citations = mutate(GOOD["answer"], GOOD["citations"])
    errors = check_citations(answer, citations, _subs())
    assert any(needle in e for e in errors), errors


def test_passage_d_une_sous_question_insuffisante_refuse():
    subs = _subs()
    subs["q3"]["passages"] = [passage("c3", "Bourse de 2000 dirhams.")]
    bad = [{"n": 1, "chunk_id": "c3", "document": "d", "page": 1, "extract": "Bourse"}]
    assert any("inventé" in e for e in check_citations("x [1]", bad, subs))


# ------------------------------------------------------------------ bout en bout
def _tool():
    return FakeSearchTool(lambda q: [passage("c1", P1, 3)] if "dur" in q.lower()
                          else [passage("c2", P2, 4)] if "doctorant" in q.lower() else [])


FABRICATED = {"answer": "La thèse dure trois ans [1] [5].",
              "citations": [{"n": 1, "chunk_id": "c_fake", "document": "X", "page": 1,
                             "extract": "inventé"}]}


def test_e2e_regeneration_corrige_les_citations():
    llm = ScriptedLLM({"generateur_final": [FABRICATED, lambda d: _heuristic(d)]})
    out = AgenticRAG(llm=llm, search_tool=_tool()).run("Quelle est la durée de la thèse ?",
                                                       "reader")
    gens = [d for t, d in llm.calls if t == "generateur_final"]
    assert len(gens) == 2 and gens[1]["citation_feedback"]           # régénération signalée
    assert gens[1]["previous"]["answer"] == FABRICATED["answer"]
    assert out["status"] == "reponse" and out["citation_feedback"] == []


def test_e2e_echec_citations_apres_G_part_en_repli():
    llm = ScriptedLLM({"generateur_final": [FABRICATED]})
    out = AgenticRAG(llm=llm, search_tool=_tool()).run("Quelle est la durée de la thèse ?",
                                                       "reader")
    assert out["generation_attempts"] == config.MAX_GENERATION_ATTEMPTS
    assert out["status"] == "je_ne_sais_pas" and out["repli_cause"] == config.CAUSE_CITATIONS
    assert out["final_answer"] == config.FALLBACK_MESSAGE and out["confidence"] == 0.0


def test_e2e_reponse_partielle_et_confiance():
    llm = ScriptedLLM({"orchestrateur": [{
        "route": "recherche", "standalone_question": "q",
        "sub_questions": [{"id": "q1", "text": "Quelle est la durée de la thèse ?"},
                          {"id": "q2", "text": "Quel est le montant de la bourse ?"}]}]})
    out = AgenticRAG(llm=llm, search_tool=_tool()).run("Durée et bourse ?", "reader")
    q1, q2 = out["sub_results"]["q1"], out["sub_results"]["q2"]
    assert q1["status"] == "valide" and q2["status"] == "insuffisant"
    assert q2["attempts"] == config.MAX_RESEARCH_ATTEMPTS               # retries épuisés
    assert out["status"] == "partielle"
    assert out["confidence"] == pytest.approx(q1["verdict"]["confidence"] * 1 / 2, abs=1e-4)
    assert config.NOT_FOUND_PREFIX in out["final_answer"]
    assert out["final_answer"].endswith(config.WARNING_MESSAGE)


def test_e2e_aucune_sous_question_valide():
    out = AgenticRAG(llm=ScriptedLLM(), search_tool=FakeSearchTool([])).run(
        "Quel est le montant de la bourse ?", "reader")
    assert out["status"] == "je_ne_sais_pas"
    assert out["repli_cause"] == config.CAUSE_NO_VALID_SUB
    assert out["final_answer"] == config.FALLBACK_MESSAGE


def test_e2e_document_restreint_meme_message_que_hors_corpus():
    """Un lecteur ne voit pas le passage réservé : même message qu'un hors corpus (F-SEC-05)."""
    def by_role_tool():
        tool = FakeSearchTool([])
        def call(name, args, *, user_role, corpora):
            tool.calls.append(user_role)
            return [passage("c1", P1, 3)] if user_role == "admin" else []
        tool.call = call
        return tool
    q = "Quelle est la durée de la thèse ?"
    admin = AgenticRAG(llm=ScriptedLLM(), search_tool=by_role_tool()).run(q, "admin")
    reader = AgenticRAG(llm=ScriptedLLM(), search_tool=by_role_tool()).run(q, "reader")
    other = AgenticRAG(llm=ScriptedLLM(), search_tool=FakeSearchTool([])).run(q, "reader")
    assert admin["status"] == "reponse"
    assert reader["final_answer"] == other["final_answer"] == config.FALLBACK_MESSAGE


@requires_data
@pytest.mark.parametrize("question,corpora", [
    ("Quelles sont les conditions d'inscription en thèse de doctorat ?", ["fstbm"]),
    ("Quel est le régime des sociétés de bourse selon le bulletin officiel ?", ["ocr"]),
])
def test_e2e_corpus_reels(question, corpora):
    from src.agent.tools import build_search_tool
    out = AgenticRAG(llm=ScriptedLLM(), search_tool=build_search_tool()).run(
        question, "reader", corpora=corpora)
    assert out["status"] in ("reponse", "partielle")
    assert out["citations"] and out["citation_feedback"] == []
    assert 0 < out["confidence"] <= 1
    nodes = [e["node"] for e in out["trace"]["events"]]
    assert nodes[:2] == ["garde_entree", "orchestrateur"] and nodes[-1] == "finalisation"


def _heuristic(data):
    from src.agent.nodes.generateur_final import heuristic
    return heuristic(data)
