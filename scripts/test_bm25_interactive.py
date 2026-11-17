
from src.retrieval.bm25 import Bm25Store
from src.retrieval.loader import load_all

# 1. Charger les corpus et initialiser BM25
store = Bm25Store(load_all())

print("===== LEGAL RAG - BM25 SEARCH =====")

# 2. Saisir la question utilisateur
query = input("\nEnter your question: ")

# 3. Choisir le corpus
corpus = input("Choose corpus (fstbm / ocr): ").strip().lower()

# 4. Saisir le rôle
role = input("Enter your role (admin / reader): ").strip().lower()

# 5. Nombre de résultats
k = int(input("Number of results (Top K): "))

# 6. Effectuer la recherche
try:
    results = store.search(query, corpus, role, k)

    # 7. Afficher les résultats
    print("\n===== SEARCH RESULTS =====")

    for i, result in enumerate(results, 1):
        print(f"\nResult {i}")
        print("Score:", round(result.score, 4))
        print("Title:", result.title)
        print("Section:", result.section)
        print("Page:", result.page_start)
        print("Text:", result.text[:500])

    if not results:
        print("No results found.")

except (ValueError, PermissionError) as e:
    print("Error:", e)
