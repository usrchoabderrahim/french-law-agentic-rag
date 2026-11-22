"""Démo interactive de la recherche avancée (multi-query + HyDE + GraphRAG + rerank).

    python -m scripts.demo_advanced
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.advanced_search import AdvancedOptions, AdvancedSearchPipeline
from src.retrieval.llm import OllamaClient
from src.retrieval.pipeline import LegalSearchPipeline


def yes(prompt: str) -> bool:
    return input(prompt + " [O/n] ").strip().lower() not in {"n", "non"}


def main() -> None:
    llm = OllamaClient()
    if not llm.available():
        print(f"(!) Ollama indisponible ou modèle '{llm.model}' absent : multi-query et HyDE seront ignorés.")
    pipeline = AdvancedSearchPipeline.from_pipeline(LegalSearchPipeline(), llm=llm)

    question = input("Question : ")
    corpus = input("Corpus (fstbm / ocr) : ").strip().lower()
    role = input("Rôle (admin / reader) : ").strip().lower()
    options = AdvancedOptions(
        multi_query=yes("Multi-query ?"), hyde=yes("HyDE ?"), graph=yes("GraphRAG ?"), rerank=yes("Re-ranking ?"))

    results, trace = pipeline.search_with_trace(question, corpus, role, k=5, options=options)

    print(f"\n===== Configuration : {options.label()} =====")
    for i, variant in enumerate(trace.reformulations, 1):
        print(f"Reformulation {i}: {variant}")
    if trace.hyde_text:
        print(f"HyDE : {trace.hyde_text[:300]}")
    print(f"Graphe : {trace.graph_seeds} entité(s) reconnue(s) dans la question")
    print(f"Candidats par classement : {trace.ranking_sizes}")
    print(f"Temps (s) : {trace.seconds}")
    for i, r in enumerate(results, 1):
        print(f"\nRésultat {i} | score {r.score:.4f} | {r.title} | p.{r.page_start} | {r.section}")
        print(r.text[:400])


if __name__ == "__main__":
    main()
