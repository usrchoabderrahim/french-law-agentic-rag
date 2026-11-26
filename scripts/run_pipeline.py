"""Interactive search: corpus, role and search type are chosen, then re-ranking is applied.
Run: python -m scripts.run_pipeline
"""
from src.retrieval.config import CORPORA, ROLES, SEARCH_MODES
from src.retrieval.pipeline import LegalSearchPipeline


def ask_choice(label: str, options) -> str:
    options = list(options)
    while True:
        value = input(f"{label} ({' / '.join(options)}): ").strip().lower()
        if value in options:
            return value
        print(f"  Invalid choice. Please choose one of: {', '.join(options)}")


pipeline = LegalSearchPipeline()

question = input("Enter your question: ").strip()
corpus = ask_choice("Choose corpus", CORPORA)
role = ask_choice("Enter role", ROLES)
mode = ask_choice("Choose search type", SEARCH_MODES)

types = pipeline.retriever.document_types(corpus)
doc_type = input(f"Document type, or Enter for all ({' / '.join(types)}): ").strip().lower() or None
if doc_type and doc_type not in types:
    print(f"  Unknown document type '{doc_type}': ignored.")
    doc_type = None

try:
    results = pipeline.search(question, corpus, role, k=5, mode=mode, document_type=doc_type)
except (ValueError, PermissionError) as error:
    raise SystemExit(f"Search refused: {error}")

print(f"\n===== FINAL RESULTS ({mode} + rerank) =====")
if not results:
    print("No result found.")
for i, result in enumerate(results, 1):
    print(f"\nResult {i}")
    print("Score:", round(result.score, 4))
    print("Title:", result.title)
    print("Section:", result.section)
    print("Page:", result.page_start)
    print("Text:", result.text[:500])