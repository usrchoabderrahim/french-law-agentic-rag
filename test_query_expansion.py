import json

from src.retrieval.llm import LLMUnavailable
from src.retrieval.query_expansion import QueryExpander, parse_query_list


class FakeLLM:
    def __init__(self, answer="", fail=False):
        self.answer, self.fail, self.prompts = answer, fail, []

    def generate(self, prompt, system=None, json_mode=False):
        self.prompts.append(prompt)
        if self.fail:
            raise LLMUnavailable("down")
        return self.answer


def test_parse_json_object_and_list():
    assert parse_query_list('{"queries": ["a", "b", "c"]}', 2) == ["a", "b"]
    assert parse_query_list('["x", "y"]', 5) == ["x", "y"]


def test_parse_plain_lines_with_bullets():
    raw = "1. Première question ?\n- Deuxième question ?\n\n* Troisième ?"
    assert parse_query_list(raw, 5) == ["Première question ?", "Deuxième question ?", "Troisième ?"]


def test_multi_query_removes_original_and_duplicates():
    answer = json.dumps({"queries": ["Quel est le délai ?", "quel est le délai", "Durée maximale ?", "Autre ?"]})
    expander = QueryExpander(FakeLLM(answer), n_queries=2)
    result = expander.multi_query("Quel est le délai ?")
    assert result == ["Durée maximale ?", "Autre ?"]


def test_multi_query_and_hyde_degrade_when_llm_is_down():
    expander = QueryExpander(FakeLLM(fail=True))
    assert expander.multi_query("question") == []
    assert expander.hyde("question") is None


def test_hyde_returns_text_and_prompt_contains_question_and_corpus():
    llm = FakeLLM("Le délai est de 72 heures.")
    text = QueryExpander(llm).hyde("Quel délai pour justifier une absence ?", corpus="fstbm")
    assert text == "Le délai est de 72 heures."
    assert "Quel délai pour justifier une absence ?" in llm.prompts[0]
    assert "FST" in llm.prompts[0]  # libellé du corpus donné au LLM
