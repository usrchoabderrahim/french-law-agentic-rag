import json

import pytest

from src.retrieval.evaluation import (
    EvalQuestion,
    compute_metrics,
    evaluate,
    first_relevant_rank,
    load_questions,
    render_markdown,
)
from tests.helpers import make_result


def test_metrics_hit_and_mrr():
    m = compute_metrics([1, 2, None, 5], ks=(1, 3, 5))
    assert m["hit@1"] == 0.25 and m["hit@3"] == 0.5 and m["hit@5"] == 0.75
    assert m["mrr"] == pytest.approx((1 + 1 / 2 + 0 + 1 / 5) / 4)


def test_mrr_ignores_ranks_beyond_cutoff_and_empty_input():
    assert compute_metrics([11], ks=(1, 10))["mrr"] == 0.0
    assert compute_metrics([], ks=(1,)) == {"hit@1": 0.0, "mrr": 0.0}


def test_relevance_by_chunk_id_and_by_keywords_with_accents_and_newlines():
    by_id = EvalQuestion("q", "?", "fstbm", relevant_chunk_ids=["b"])
    assert first_relevant_rank([make_result("a"), make_result("b")], by_id) == 2
    by_kw = EvalQuestion("q", "?", "fstbm", relevant_keywords=["délai", "72 h"])
    result = make_result("x")
    result.text = "Le DELAI\nest de 72 h."
    assert by_kw.is_relevant(result) and first_relevant_rank([make_result("z")], by_kw) is None


def test_evaluate_and_markdown_report():
    qs = [EvalQuestion("q1", "a", "fstbm", relevant_chunk_ids=["t1"]),
          EvalQuestion("q2", "b", "ocr", relevant_chunk_ids=["t2"])]
    answers = {"q1": ["t1", "x"], "q2": ["x", "y", "t2"]}
    report = evaluate("demo", lambda q, k: [make_result(c) for c in answers[q.id]][:k], qs, k=5, ks=(1, 3))
    assert report.metrics["hit@1"] == 0.5 and report.metrics["hit@3"] == 1.0
    assert report.by_corpus["fstbm"]["mrr"] == 1.0
    md = render_markdown([report], ks=(1, 3))
    assert "demo" in md and "hit@3" in md and "`ocr`" in md


def test_load_questions_validates(tmp_path):
    good = tmp_path / "q.jsonl"
    good.write_text(json.dumps({"question": "?", "corpus": "fstbm", "relevant_keywords": ["a"]}) + "\n# commentaire\n")
    assert load_questions(good)[0].role == "reader"
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"question": "?", "corpus": "fstbm"}) + "\n")
    with pytest.raises(ValueError):
        load_questions(bad)


def test_shipped_question_set_matches_the_data():
    from src.retrieval.loader import load_all
    from src.retrieval.models import SearchResult

    corpora = load_all()
    for q in load_questions("data/eval/questions.jsonl"):
        assert any(
            q.is_relevant(SearchResult(c.chunk_id, c.text, 0, "x", q.corpus, "", None, None, None, None, None, None, None))
            for c in corpora[q.corpus]
        ), f"{q.id}: aucun chunk pertinent dans le corpus"
