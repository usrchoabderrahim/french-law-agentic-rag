"""Real test of method 3 (hybrid = Chroma + BM25 + RRF) and comparison of the 3 modes.
Run: python -m scripts.check_hybrid
"""
from scripts.check_utils import check, show, summary
from src.retrieval.search import Retriever
from src.retrieval.vector_store import ChromaStore

store = ChromaStore()
store.build_all()
retriever = Retriever(vector_store=store)

question = "inscription pédagogique suspendue pour un étudiant absent au semestre précédent"
for mode in ("vector", "bm25", "hybrid"):
    print(f"--- FST | {mode} ---")
    res = retriever.search(question, "fstbm", "reader", k=5, mode=mode)
    show(res)
    check(f"FST {mode}: Article 5.1 in top 5", "Article 5.1" in [r.section for r in res])
    check(f"FST {mode}: method label is '{mode}'", all(r.method == mode for r in res))

for mode in ("vector", "bm25", "hybrid"):
    print(f"--- OCR | {mode} ---")
    res = retriever.search("dahir portant organisation d'un service public", "ocr", "reader", k=5, mode=mode)
    show(res)
    check(f"OCR {mode}: results found", len(res) > 0)
    check(f"OCR {mode}: only bulletin_officiel", all(r.document_type == "bulletin_officiel" for r in res))

print("--- hybrid + document type + role ---")
res = retriever.search("inscription", "fstbm", "reader", k=5, document_type="guide")
show(res)
check("hybrid filter guide: only guides", len(res) > 0 and all(r.document_type == "guide" for r in res))
res = retriever.search("inscription", "fstbm", "admin", k=3)
check("hybrid works for admin", len(res) > 0)
for bad in (None, "lecteur"):
    try:
        retriever.search("inscription", "fstbm", bad)
        check(f"hybrid role {bad!r} rejected", False)
    except PermissionError:
        check(f"hybrid role {bad!r} rejected", True)

summary()
