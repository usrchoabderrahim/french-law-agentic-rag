"""Usage (from project root): python -m scripts.demo_search"""
from src.retrieval.search import Retriever
from src.retrieval.vector_store import ChromaStore

store = ChromaStore()
print("Indexed:", store.build_all())  # {'fstbm': 44, 'ocr': 981}

retriever = Retriever(vector_store=store)
print("Document types (fstbm):", retriever.document_types("fstbm"))

QUESTION = "conditions d'inscription pédagogique"
for mode in ("vector", "bm25", "hybrid"):
    print(f"\n=== {mode} | corpus=fstbm | role=reader ===")
    for r in retriever.search(QUESTION, "fstbm", "reader", k=3, mode=mode):
        print(f"{r.score:.3f} | p.{r.page_start} | {r.section} | {r.text[:70]!r}")

print("\n=== hybrid + document_type='guide' ===")
for r in retriever.search("inscription", "fstbm", "reader", k=3, document_type="guide"):
    print(f"{r.score:.4f} | {r.document_type} | {r.title[:50]}")



























# """Usage (from project root): python -m scripts.demo_search"""
# from src.retrieval.search import Retriever
# from src.retrieval.vector_store import ChromaStore

# store = ChromaStore()
# print("Indexed:", store.build_all())  # {'fstbm': 44, 'ocr': 981}

# retriever = Retriever(vector_store=store)
# print("Document types (fstbm):", retriever.document_types("fstbm"))

# QUESTION = "conditions d'inscription pédagogique"
# for mode in ("vector", "bm25", "hybrid"):
#     print(f"\n=== {mode} | corpus=fstbm | role=lecteur ===")
#     for r in retriever.search(QUESTION, "fstbm", "lecteur", k=3, mode=mode):
#         print(f"{r.score:.3f} | p.{r.page_start} | {r.section} | {r.text[:70]!r}")

# print("\n=== hybrid + document_type='guide' ===")
# for r in retriever.search("inscription", "fstbm", "lecteur", k=3, document_type="guide"):
#     print(f"{r.score:.4f} | {r.document_type} | {r.title[:50]}")
