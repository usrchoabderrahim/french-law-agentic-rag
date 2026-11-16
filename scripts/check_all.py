"""GLOBAL test: 2 corpora x 3 methods x 2 roles + filters + errors.
Run: python -m scripts.check_all
"""
import itertools

from scripts.check_utils import check, summary
from src.retrieval.config import CORPORA, ROLES, SEARCH_MODES
from src.retrieval.loader import load_all
from src.retrieval.search import Retriever
from src.retrieval.vector_store import ChromaStore

store = ChromaStore()
counts = store.build_all()
check("both corpora indexed", counts == {"fstbm": 44, "ocr": 981}, str(counts))

retriever = Retriever(vector_store=store)
questions = {
    "fstbm": "inscription pédagogique suspendue pour un étudiant absent au semestre précédent",
    "ocr": "dahir portant organisation d'un service public",
}

# 1. every combination works and stays inside the chosen corpus
for corpus, mode, role in itertools.product(CORPORA, SEARCH_MODES, ROLES):
    res = retriever.search(questions[corpus], corpus, role, k=5, mode=mode)
    check(f"{corpus} | {mode:6s} | {role:6s}: results from the right corpus",
          len(res) > 0 and all(r.corpus == corpus for r in res), f"{len(res)} results")

# 2. document type filter on every method
for mode in SEARCH_MODES:
    types = retriever.document_types("fstbm")
    for dtype in types:
        res = retriever.search("étudiant inscription", "fstbm", "reader", k=5, mode=mode, document_type=dtype)
        check(f"filter {dtype} | {mode}", all(r.document_type == dtype for r in res), f"{len(res)} results")

# 3. a restricted chunk must never be returned to a role that cannot see it
allowed = {c.chunk_id: c.allowed_roles for chunks in load_all().values() for c in chunks}
for mode, role in itertools.product(SEARCH_MODES, ROLES):
    res = retriever.search(questions["fstbm"], "fstbm", role, k=10, mode=mode)
    check(f"no leak | {mode} | {role}", all(role in allowed[r.chunk_id] for r in res))

# 4. errors are refused
for label, kwargs, error in [
    ("no role", dict(role=None), PermissionError),
    ("unknown role", dict(role="lecteur"), PermissionError),
    ("unknown corpus", dict(corpus="xyz"), ValueError),
    ("unknown mode", dict(mode="magic"), ValueError),
    ("empty question", dict(query="  "), ValueError),
]:
    args = dict(query="inscription", corpus="fstbm", role="reader") | kwargs
    try:
        retriever.search(**args)
        check(f"refused: {label}", False)
    except error:
        check(f"refused: {label}", True)

summary()
