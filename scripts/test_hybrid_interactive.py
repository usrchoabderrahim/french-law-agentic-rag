
from src.retrieval.search import Retriever
from src.retrieval.vector_store import ChromaStore

print("===== LEGAL RAG - HYBRID SEARCH =====")

# Initialisation
store = ChromaStore()
store.build_all()

retriever = Retriever(vector_store=store)

# Saisie de la question
question = input("\nEnter your question: ")

corpus = input("Choose corpus (fstbm / ocr): ").strip().lower()
role = input("Enter your role (admin / reader): ").strip().lower()

# Nombre fixe de résultats
k = 5

# Comparaison des trois méthodes
for mode in ("vector", "bm25", "hybrid"):

    print(f"\n========== {mode.upper()} ==========")

    try:
        results = retriever.search(
            question,
            corpus,
            role,
            k=k,
            mode=mode
        )

        if not results:
            print("No results found.")
            continue

        for i, result in enumerate(results, 1):
            print(f"\nResult {i}")
            print("Score:", round(result.score, 4))
            print("Title:", result.title)
            print("Section:", result.section)
            print("Page:", result.page_start)
            print("Document type:", result.document_type)
            print("Text:", result.text[:500])

    except (ValueError, PermissionError) as e:
        print("Error:", e)
        break
