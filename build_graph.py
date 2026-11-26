"""Construit le graphe léger de chaque corpus et l'exporte (inspection + rapport).

    python -m scripts.build_graph                 # extraction par règles (rapide, gratuit)
    python -m scripts.build_graph --llm           # + entités/relations extraites par Ollama (lent)
    python -m scripts.build_graph --role admin
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.advanced_config import GRAPH_DIR
from src.retrieval.graph_rag import GraphStore, LLMExtractor, RuleExtractor, corpus_vocabulary
from src.retrieval.loader import load_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default="reader", choices=["reader", "admin"])
    parser.add_argument("--llm", action="store_true", help="extraction LLM (Ollama) en plus des règles")
    args = parser.parse_args()

    factory = None
    if args.llm:
        from src.retrieval.llm import OllamaClient

        llm = OllamaClient()
        if not llm.available():
            sys.exit(f"Ollama indisponible ou modèle '{llm.model}' absent (ollama pull {llm.model}).")
        factory = lambda chunks: LLMExtractor(llm, RuleExtractor(corpus_vocabulary(chunks)))  # noqa: E731

    corpora = load_all()
    store = GraphStore(corpora, extractor_factory=factory)
    for name in corpora:
        graph = store.graph(name, args.role)
        out = GRAPH_DIR / f"{name}_{args.role}.json"
        graph.save(out)
        stats = graph.stats()
        print(f"\n=== {name} (rôle {args.role}) -> {out}")
        for key in ("chunks", "entities", "entities_by_type", "typed_relations",
                    "cooccurrence_edges", "chunks_without_entity"):
            print(f"  {key}: {stats[key]}")
        print("  entités les plus citées:", stats["top_entities"][:5])


if __name__ == "__main__":
    main()
