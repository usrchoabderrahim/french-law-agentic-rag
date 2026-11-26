
from src.retrieval.query_encoder import QueryEncoder
from src.retrieval.vector_store import ChromaStore

print("===== LEGAL RAG - VECTOR SEARCH =====")

# 1. Initialiser ChromaDB
store = ChromaStore()

# 2. Vérifier / construire les index
counts = store.build_all()
print("\nIndexed collections:", counts)

# 3. Initialiser BGE-M3
encoder = QueryEncoder()

# 4. Saisir la question de l'utilisateur
query = input("\nEnter your question: ")

# 5. Choisir le corpus
corpus = input("Choose corpus (fstbm / ocr): ").strip().lower()

# 6. Saisir le rôle
role = input("Enter your role (admin / reader): ").strip().lower()

# 7. Nombre de résultats
k = int(input("Number of results (Top K): "))

try:
    # 8. Transformer la question en embedding
    query_vector = encoder.encode(query)

    print("\nQuestion encoded successfully!")
    print("Vector dimensions:", len(query_vector))

    # 9. Effectuer la recherche vectorielle
    results = store.search(
        query_vector,
        corpus,
        role,
        k
    )

    # 10. Afficher les résultats
    print("\n===== VECTOR SEARCH RESULTS =====")

    for i, result in enumerate(results, 1):
        print(f"\nResult {i}")
        print("Score:", round(result.score, 4))
        print("Title:", result.title)
        print("Section:", result.section)
        print("Page:", result.page_start)
        print("Document type:", result.document_type)
        print("Text:", result.text[:500])

    if not results:
        print("No results found.")

except (ValueError, PermissionError) as e:
    print("\nError:", e)
