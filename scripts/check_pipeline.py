
from src.retrieval.pipeline import LegalSearchPipeline

pipeline = LegalSearchPipeline()

question = input("Enter your question: ")
corpus = input("Choose corpus (fstbm / ocr): ").strip().lower()
role = input("Enter role (admin / reader): ").strip().lower()

results = pipeline.search(
    question,
    corpus,
    role,
    k=5
)

print("\n===== FINAL LEGAL SEARCH RESULTS =====")

for i, result in enumerate(results, 1):
    print(f"\nResult {i}")
    print("Score:", round(result.score, 4))
    print("Title:", result.title)
    print("Section:", result.section)
    print("Page:", result.page_start)
    print("Text:", result.text[:500])
