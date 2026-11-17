"""Real test of re-ranking after each search method (vector, bm25, hybrid).
Run: python -m scripts.check_reranking
"""
from scripts.check_utils import check, summary
from src.retrieval.pipeline import LegalSearchPipeline

pipeline = LegalSearchPipeline()

QUESTIONS = {
    "fstbm": "inscription pédagogique suspendue pour un étudiant absent au semestre précédent",
    "ocr": "dahir portant organisation d'un service public",
}


def show(results):
    for i, r in enumerate(results, 1):
        print(f"   {i}. score={r.score:.4f} | p.{r.page_start} | {str(r.section)[:28]:28s} | {r.text[:60]!r}")


for corpus, question in QUESTIONS.items():
    for mode in ("vector", "bm25", "hybrid"):
        print(f"\n--- {corpus} | {mode} + rerank ---")
        before = pipeline.search(question, corpus, "reader", k=5, mode=mode, rerank=False)
        after = pipeline.search(question, corpus, "reader", k=5, mode=mode, rerank=True)
        print("  before rerank:")
        show(before)
        print("  after rerank:")
        show(after)

        check(f"{corpus} {mode}: results returned", 0 < len(after) <= 5)
        check(f"{corpus} {mode}: labelled 'rerank'", all(r.method == "rerank" for r in after))
        check(f"{corpus} {mode}: sorted by score", [r.score for r in after] == sorted((r.score for r in after), reverse=True))
        check(f"{corpus} {mode}: stays in corpus", all(r.corpus == corpus for r in after))
        if corpus == "fstbm":
            check(f"fstbm {mode}: best result is Article 5.1", after[0].section == "Article 5.1")

print("\n--- filters and roles through the pipeline ---")
res = pipeline.search("inscription", "fstbm", "reader", k=5, mode="hybrid", document_type="guide")
check("document type 'guide' respected after rerank", len(res) > 0 and all(r.document_type == "guide" for r in res))
check("admin can search", len(pipeline.search("inscription", "fstbm", "admin", k=3)) > 0)
for bad in (None, "lecteur"):
    try:
        pipeline.search("inscription", "fstbm", bad)
        check(f"role {bad!r} rejected", False)
    except PermissionError:
        check(f"role {bad!r} rejected", True)

summary()