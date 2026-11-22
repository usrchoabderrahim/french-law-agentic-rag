"""Évaluation de la recherche : hit rate@k et MRR sur un jeu de questions.

Format du jeu de questions (une ligne JSON par question, data/eval/questions.jsonl) :

    {"id": "q01", "question": "...", "corpus": "fstbm", "role": "reader",
     "relevant_chunk_ids": ["8d8d39..."],            # option 1 : ids exacts
     "relevant_keywords": ["72 h", "justifi"]}        # option 2 : un chunk est pertinent s'il
                                                      #            contient TOUS ces mots-clés

- hit@k  : part des questions où un chunk pertinent est dans les k premiers résultats.
- MRR@K  : moyenne de 1/rang du premier chunk pertinent (0 s'il n'est pas dans le top K).
"""
from __future__ import annotations

import json
import statistics
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .advanced_config import EVAL_KS
from .models import SearchResult


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return " ".join(text.replace("’", "'").split())  # espaces/retours ligne OCR normalisés


@dataclass
class EvalQuestion:
    id: str
    question: str
    corpus: str
    role: str = "reader"
    relevant_chunk_ids: list[str] = field(default_factory=list)
    relevant_keywords: list[str] = field(default_factory=list)

    def is_relevant(self, result: SearchResult) -> bool:
        if result.chunk_id in self.relevant_chunk_ids:
            return True
        if self.relevant_keywords:
            text = _fold(result.text)
            return all(_fold(kw) in text for kw in self.relevant_keywords)
        return False


def load_questions(path: Path) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            row = json.loads(line)
            q = EvalQuestion(
                id=row.get("id", f"q{line_no:03d}"),
                question=row["question"],
                corpus=row["corpus"],
                role=row.get("role", "reader"),
                relevant_chunk_ids=row.get("relevant_chunk_ids", []),
                relevant_keywords=row.get("relevant_keywords", []),
            )
            if not q.relevant_chunk_ids and not q.relevant_keywords:
                raise ValueError(f"{q.id}: ni relevant_chunk_ids ni relevant_keywords")
            questions.append(q)
    return questions


def first_relevant_rank(results: Iterable[SearchResult], question: EvalQuestion) -> int | None:
    """Rang (à partir de 1) du premier résultat pertinent, None s'il n'y en a pas."""
    for rank, result in enumerate(results, start=1):
        if question.is_relevant(result):
            return rank
    return None


def compute_metrics(ranks: list[int | None], ks: tuple[int, ...] = EVAL_KS, mrr_k: int | None = None) -> dict[str, float]:
    """hit@k pour chaque k et MRR (tronqué à mrr_k, par défaut le plus grand k)."""
    n = len(ranks)
    if n == 0:
        return {**{f"hit@{k}": 0.0 for k in ks}, "mrr": 0.0}
    mrr_k = mrr_k or max(ks)
    metrics = {f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in ks}
    metrics["mrr"] = sum(1.0 / r for r in ranks if r is not None and r <= mrr_k) / n
    return metrics


@dataclass
class EvalReport:
    name: str
    n_questions: int
    metrics: dict[str, float]
    by_corpus: dict[str, dict[str, float]]
    avg_seconds: float
    details: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


SearchFn = Callable[[EvalQuestion, int], list[SearchResult]]


def evaluate(
    name: str,
    search_fn: SearchFn,
    questions: list[EvalQuestion],
    k: int = max(EVAL_KS),
    ks: tuple[int, ...] = EVAL_KS,
) -> EvalReport:
    """`search_fn(question, k)` doit renvoyer les k meilleurs résultats."""
    ranks: list[int | None] = []
    per_corpus: dict[str, list[int | None]] = {}
    details, durations = [], []
    for q in questions:
        start = time.perf_counter()
        results = search_fn(q, k)
        durations.append(time.perf_counter() - start)
        rank = first_relevant_rank(results, q)
        ranks.append(rank)
        per_corpus.setdefault(q.corpus, []).append(rank)
        details.append({"id": q.id, "corpus": q.corpus, "question": q.question, "rank": rank,
                        "top_chunk_ids": [r.chunk_id for r in results[:3]]})
    return EvalReport(
        name=name,
        n_questions=len(questions),
        metrics=compute_metrics(ranks, ks, mrr_k=k),
        by_corpus={c: compute_metrics(r, ks, mrr_k=k) for c, r in per_corpus.items()},
        avg_seconds=statistics.fmean(durations) if durations else 0.0,
        details=details,
    )


def render_markdown(reports: list[EvalReport], ks: tuple[int, ...] = EVAL_KS) -> str:
    """Tableau comparatif prêt à coller dans le rapport."""
    header = "| Configuration | " + " | ".join(f"hit@{k}" for k in ks) + " | MRR | s/question |"
    lines = [header, "|" + "---|" * (len(ks) + 3)]
    for r in reports:
        cells = " | ".join(f"{r.metrics[f'hit@{k}']:.3f}" for k in ks)
        lines.append(f"| {r.name} | {cells} | {r.metrics['mrr']:.3f} | {r.avg_seconds:.2f} |")
    n = reports[0].n_questions if reports else 0
    out = [f"Questions évaluées : {n}\n", *lines]
    corpora = sorted({c for r in reports for c in r.by_corpus})
    for corpus in corpora:
        out += ["", f"**Corpus `{corpus}`**", "", header.replace("| s/question |", "|"), "|" + "---|" * (len(ks) + 2)]
        for r in reports:
            if corpus in r.by_corpus:
                m = r.by_corpus[corpus]
                cells = " | ".join(f"{m[f'hit@{k}']:.3f}" for k in ks)
                out.append(f"| {r.name} | {cells} | {m['mrr']:.3f} |")
    return "\n".join(out) + "\n"
