"""Central configuration for the retrieval layer."""
from __future__ import annotations

from pathlib import Path

DATA_DIR = Path("data/embeddings")

# corpus name -> folder containing all_chunks.jsonl, embeddings.npy, embedding_ids.jsonl
CORPORA: dict[str, Path] = {
    "fstbm": DATA_DIR / "embeddings_fstbm",
    "ocr": DATA_DIR / "embeddings_ocr",
}

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# Roles exactly as stored in the ingestion data (allowed_roles): no rename, no addition.
ROLES: tuple[str, ...] = ("admin", "reader")

CHROMA_DIR = Path("data/chroma")
# One separate Chroma collection per corpus: the user picks the corpus in the UI.
COLLECTIONS: dict[str, str] = {"fstbm": "ma_fstbm", "ocr": "ma_ocr"}

# Labels shown in the front end.
CORPUS_LABELS: dict[str, str] = {
    "fstbm": "FST Béni Mellal (règlements, guides)",
    "ocr": "Bulletins officiels (PDF scannés, OCR)",
}

# Search settings
DEFAULT_TOP_K = 5
CANDIDATES_PER_METHOD = 20  # candidates taken from each method before fusion
RRF_K = 60                  # standard constant of Reciprocal Rank Fusion
SEARCH_MODES = ("vector", "bm25", "hybrid")






































# """Central configuration for the retrieval layer."""
# from __future__ import annotations

# from pathlib import Path

# DATA_DIR = Path("data/embeddings")

# # corpus name -> folder containing all_chunks.jsonl, embeddings.npy, embedding_ids.jsonl
# CORPORA: dict[str, Path] = {
#     "fstbm": DATA_DIR / "embeddings_fstbm",
#     "ocr": DATA_DIR / "embeddings_ocr",
# }

# EMBEDDING_MODEL = "BAAI/bge-m3"
# EMBEDDING_DIM = 1024

# # Roles required by the spec (admin, editeur, lecteur).
# # The ingestion team used English names: map them to the spec names.
# ROLE_ALIASES: dict[str, str] = {
#     "reader": "lecteur",
#     "user": "lecteur",
#     "editor": "editeur",
#     "admin": "admin",
# }

# CHROMA_DIR = Path("data/chroma")
# # One separate Chroma collection per corpus: the user picks the corpus in the UI.
# COLLECTIONS: dict[str, str] = {"fstbm": "ma_fstbm", "ocr": "ma_ocr"}

# # Labels shown in the front end.
# CORPUS_LABELS: dict[str, str] = {
#     "fstbm": "FST Béni Mellal (règlements, guides)",
#     "ocr": "Bulletins officiels (PDF scannés, OCR)",
# }

# # Search settings
# DEFAULT_TOP_K = 5
# CANDIDATES_PER_METHOD = 20  # candidates taken from each method before fusion
# RRF_K = 60                  # standard constant of Reciprocal Rank Fusion
# SEARCH_MODES = ("vector", "bm25", "hybrid")
