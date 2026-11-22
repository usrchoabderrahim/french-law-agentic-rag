"""Compare les configurations de recherche (A vs B) : hit rate@k et MRR.

    python -m scripts.run_eval                              # toutes les configurations
    python -m scripts.run_eval --configs base,rerank,B_full # un sous-ensemble
    python -m scripts.run_eval --questions data/eval/questions.jsonl data/eval/questions_generated.jsonl

Sorties : results/eval_B_<date>.json (détail par question) et reports/eval_B.md (tableau).
Les appels LLM sont mis en cache (data/cache/llm_cache.json) : la 2e exécution est quasi instantanée.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.advanced_config import EVAL_KS, EVAL_QUESTIONS, REPORTS_DIR, RESULTS_DIR
from src.retrieval.advanced_search import AdvancedOptions, AdvancedSearchPipeline
from src.retrieval.evaluation import EvalQuestion, evaluate, load_questions, render_markdown

O = AdvancedOptions
# nom court -> (libellé dans le tableau, options)
CONFIGS: dict[str, tuple[str, AdvancedOptions]] = {
    "base":      ("A : hybride",                        O(False, False, False, rerank=False)),
    "rerank":    ("A : hybride + rerank",               O(False, False, False, rerank=True)),
    "multi":     ("B : + multi-query",                  O(True, False, False, rerank=False)),
    "hyde":      ("B : + HyDE",                         O(False, True, False, rerank=False)),
    "graph":     ("B : + GraphRAG",                     O(False, False, True, rerank=False)),
    "B_nogr":    ("B : multi + HyDE",                   O(True, True, False, rerank=False)),
    "B_all":     ("B : multi + HyDE + graphe",          O(True, True, True, rerank=False)),
    "B_full":    ("B complet : tout + rerank",          O(True, True, True, rerank=True)),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", nargs="+", type=Path, default=[EVAL_QUESTIONS])
    parser.add_argument("--configs", default=",".join(CONFIGS), help=f"parmi: {', '.join(CONFIGS)}")
    parser.add_argument("--k", type=int, default=max(EVAL_KS))
    args = parser.parse_args()

    names = [n.strip() for n in args.configs.split(",") if n.strip()]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        sys.exit(f"Configurations inconnues: {unknown}. Choix: {list(CONFIGS)}")
    questions = [q for path in args.questions for q in load_questions(path)]
    print(f"{len(questions)} questions, configurations: {names}")

    from src.retrieval.llm import OllamaClient
    from src.retrieval.pipeline import LegalSearchPipeline

    llm = OllamaClient()
    needs_llm = any(CONFIGS[n][1].multi_query or CONFIGS[n][1].hyde for n in names)
    if needs_llm and not llm.available():
        sys.exit(f"Ollama indisponible ou modèle '{llm.model}' absent (ollama pull {llm.model}).")

    base_pipeline = LegalSearchPipeline()  # index Chroma + BM25 + reranker de la sous-tâche A
    pipeline = AdvancedSearchPipeline.from_pipeline(base_pipeline, llm=llm)

    reports = []
    for name in names:
        label, options = CONFIGS[name]
        print(f"\n--- {label}")

        def search(q: EvalQuestion, k: int, options=options):
            return pipeline.search(q.question, q.corpus, q.role, k=k, options=options)

        report = evaluate(label, search, questions, k=args.k)
        reports.append(report)
        print("   ", {k: round(v, 3) for k, v in report.metrics.items()}, f"{report.avg_seconds:.2f}s/question")

    RESULTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (RESULTS_DIR / f"eval_B_{stamp}.json").write_text(
        json.dumps({"model": llm.model, "k": args.k, "reports": [r.to_dict() for r in reports]},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    markdown = render_markdown(reports)
    (REPORTS_DIR / "eval_B.md").write_text(
        f"# Évaluation sous-tâche B ({stamp}, LLM : {llm.model})\n\n{markdown}", encoding="utf-8")
    print("\n" + markdown)
    print(f"Écrit : results/eval_B_{stamp}.json, reports/eval_B.md")


if __name__ == "__main__":
    main()
