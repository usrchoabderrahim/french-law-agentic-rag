"""Real test of method 2 (BM25) on both corpora. No model, no Chroma needed.
Run: python -m scripts.check_bm25
"""
from scripts.check_utils import check, show, summary
from src.retrieval.bm25 import Bm25Store
from src.retrieval.loader import load_all

store = Bm25Store(load_all())

print("--- FST corpus ---")


# la question de user
q = "inscription pédagogique étudiant absent semestre précédent"
res = store.search(q, "fstbm", "reader", 5)
show(res)

# les test
check("FST: Article 5.1 in top 5", "Article 5.1" in [r.section for r in res])

#autre test
check("FST: all results come from fstbm", all(r.corpus == "fstbm" for r in res))

print("--- OCR corpus ---")
res = store.search("dahir portant", "ocr", "reader", 5)
show(res)

#? result men OCR
check("OCR: results found", len(res) > 0)

  #, TOUS TYPE bulletin_officiel
check("OCR: all are bulletin_officiel", all(r.document_type == "bulletin_officiel" for r in res))

print("--- roles ---")

#, ADMIN CAN RECHERCHE
check("admin can search", len(store.search(q, "fstbm", "admin", 5)) > 0)

# LES RLE NON AUTORISER REJECTED
for bad in (None, "hacker", "lecteur"):
    try:
        store.search(q, "fstbm", bad, 5)
        check(f"role {bad!r} rejected", False)
    except PermissionError:
        check(f"role {bad!r} rejected", True)

print("--- document type filter ---")

# FILTER TYPE RESULT
res = store.search("inscription", "fstbm", "reader", 10, {"document_type": "guide"})
check("filter guide: only guides", len(res) > 0 and all(r.document_type == "guide" for r in res))

summary()
