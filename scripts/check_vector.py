"""Real test of method 1 (ChromaDB + bge-m3 on the question).
Needs: pip install chromadb sentence-transformers   (first run downloads bge-m3)
Run: python -m scripts.check_vector
"""
from scripts.check_utils import check, show, summary
from src.retrieval.query_encoder import QueryEncoder
from src.retrieval.vector_store import ChromaStore

store = ChromaStore()
counts = store.build_all()
print("Indexed in Chroma:", counts)
check("FST indexed: 44", counts["fstbm"] == 44)
check("OCR indexed: 981", counts["ocr"] == 981)

encoder = QueryEncoder()
vector = encoder.encode("inscription pédagogique suspendue pour un étudiant absent au semestre précédent")
check("question vector has 1024 numbers", len(vector) == 1024)

print("--- FST corpus ---")
res = store.search(vector, "fstbm", "reader", 5)
show(res)
check("FST: Article 5.1 in top 5", "Article 5.1" in [r.section for r in res])
check("FST: scores sorted high -> low", [r.score for r in res] == sorted((r.score for r in res), reverse=True))

print("--- OCR corpus ---")
res = store.search(encoder.encode("dahir portant organisation d'un service public"), "ocr", "reader", 5)
show(res)
check("OCR: results found", len(res) == 5)
check("OCR: all are bulletin_officiel", all(r.document_type == "bulletin_officiel" for r in res))

print("--- roles and filters ---")
check("admin can search", len(store.search(vector, "fstbm", "admin", 3)) == 3)
for bad in (None, "hacker", "lecteur"):
    try:
        store.search(vector, "fstbm", bad, 3)
        check(f"role {bad!r} rejected", False)
    except PermissionError:
        check(f"role {bad!r} rejected", True)
res = store.search(vector, "fstbm", "reader", 5, {"document_type": "charte"})
check("filter charte: only chartes", len(res) > 0 and all(r.document_type == "charte" for r in res))

summary()
