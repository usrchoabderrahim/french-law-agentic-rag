
from src.retrieval.search import Retriever
from src.retrieval.vector_store import ChromaStore
from src.retrieval.reranker import Reranker

store = ChromaStore()
store.build_all()

retriever = Retriever(vector_store=store)
reranker = Reranker()

question = "inscription pédagogique suspendue pour un étudiant absent au semestre précédent"

# Première étape : recherche hybride
results = retriever.search(
    question,
    "fstbm",
    "reader",
    k=20,
    mode="hybrid"
)

# Deuxième étape : reranking
final_results = reranker.rerank(
    question,
    results,
    k=5
)

# Affichage
print("\n===== RERANKING RESULTS =====")

for i, result in enumerate(final_results, 1):
    print(f"\nResult {i}")
    print("Score:", round(result.score, 4))
    print("Title:", result.title)
    print("Section:", result.section)
    print("Page:", result.page_start)
    print("Text:", result.text[:500])
