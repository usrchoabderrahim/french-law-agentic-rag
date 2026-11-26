"""Configuration de la sous-tâche B (recherche avancée). Ne modifie pas config.py."""
from __future__ import annotations

import os
from pathlib import Path

# --- LLM local (Ollama) -------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")  # ollama pull qwen2.5:7b
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))  # secondes
LLM_TEMPERATURE = 0.2
LLM_CACHE_PATH = Path("data/cache/llm_cache.json")  # rend l'évaluation rapide et reproductible

# --- Expansion de requête (F-RET-05) ------------------------------------------
MULTI_QUERY_COUNT = 3

# --- Fusion : poids de chaque classement dans le RRF --------------------------
WEIGHT_ORIGINAL = 1.0
WEIGHT_MULTI_QUERY = 1.0
WEIGHT_HYDE = 1.0
WEIGHT_GRAPH = 0.5  # le graphe est un signal d'appoint, pas la source principale

# --- GraphRAG léger (F-RET-06) --------------------------------------------------
GRAPH_DIR = Path("data/indexes/graph")
GRAPH_NEIGHBOR_LIMIT = 10     # nb max de voisins "co-occurrence" retenus par entité
GRAPH_WEIGHT_SEED = 1.0       # entité trouvée dans la question
GRAPH_WEIGHT_RELATION = 0.6   # voisin via relation typée (modifie, abroge...)
GRAPH_WEIGHT_COOCCUR = 0.3    # voisin par co-occurrence dans un même chunk

# --- Évaluation -----------------------------------------------------------------
EVAL_DIR = Path("data/eval")
EVAL_QUESTIONS = EVAL_DIR / "questions.jsonl"
RESULTS_DIR = Path("results")
REPORTS_DIR = Path("reports")
EVAL_KS = (1, 3, 5, 10)
