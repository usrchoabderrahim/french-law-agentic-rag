"""Interroger l'assistant agentique en ligne de commande.

Usage (depuis la racine du dépôt) :
    python -m scripts.ask_agent "Quelle est la durée de la thèse ?" --role reader
    python -m scripts.ask_agent --role admin --corpus ocr          # mode conversation
    python -m scripts.ask_agent "..." --role reader --trace        # trace pour l'admin

Variables : RAG_LLM=auto|ollama|heuristic, RAG_SEARCH_MODE=auto|bm25, OLLAMA_MODEL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src.agent import AgenticRAG
from src.agent.llm import get_llm


def load_env(path: Path = Path(".env")) -> None:
    """Lit un fichier .env local (ignoré par Git) ; l'environnement réel reste prioritaire."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def show(state: dict, trace: bool) -> None:
    print(f"\n{state['final_answer']}\n")
    conf = state.get("confidence")
    print(f"statut : {state['status']} | confiance : {'—' if conf is None else f'{conf:.2f}'}")
    for c in state.get("citations") or []:
        print(f"  [{c.get('n')}] {c.get('document')} — p. {c.get('page')} : "
              f"« {str(c.get('extract'))[:120]} »")
    if state.get("repli_cause") == "llm_indisponible":  # info pour l'opérateur du CLI
        detail = next((e.get("detail") for e in reversed(state["trace"]["events"])
                       if e.get("erreur") == "llm_indisponible"), "")
        print(f"(LLM indisponible : {detail} — la réponse n'a pas pu être cherchée)")
    if trace:
        print(json.dumps(state["trace"], ensure_ascii=False, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Assistant juridique agentique (8 nœuds)")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--role", help="admin | reader (obligatoire, sinon refus T-14)")
    parser.add_argument("--corpus", nargs="+", default=["fstbm", "ocr"], choices=["fstbm", "ocr"])
    parser.add_argument("--llm", default=None, help="auto | ollama | heuristic")
    parser.add_argument("--trace", action="store_true", help="affiche la trace (administrateurs)")
    args = parser.parse_args()
    load_env()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    llm = get_llm(args.llm)
    rag = AgenticRAG(llm=llm)
    print(f"LLM : {type(llm).__name__} | corpus : {', '.join(args.corpus)}")
    if args.question:
        show(rag.run(args.question, args.role, corpora=args.corpus), args.trace)
        return
    history: list[dict] = []
    while True:
        try:
            question = input("\nQuestion (vide pour quitter) > ").strip()
        except EOFError:
            break
        if not question:
            break
        state = rag.run(question, args.role, history=history, corpora=args.corpus)
        show(state, args.trace)
        history += [{"role": "user", "content": question},
                    {"role": "assistant", "content": state["final_answer"],
                     "citations": state.get("citations") or []}]


if __name__ == "__main__":
    main()
