"""Paramètres du workflow agentique (seuils, limites, messages) — source unique.

Les valeurs suivent `task_rag_fr.docx`. Celles que le document ne fixe pas
(N, G, budgets) sont choisies ici et documentées dans docs/AGENTIC_RAG.md.
"""
from __future__ import annotations

# Rôles tels qu'ils existent dans les données d'ingestion (voir src/retrieval/config.py).
from src.retrieval.config import ROLES  # ("admin", "reader")

# --- garde_entree -------------------------------------------------------------
MAX_QUESTION_CHARS = 2000
INJECTION_PATTERNS: tuple[str, ...] = (
    r"ignore[rz]?\s+(tes|vos|les|toutes?\s+les)\s+(instructions|consignes|r[eè]gles)",
    r"oublie[rz]?\s+(tes|vos|les)\s+(instructions|consignes|r[eè]gles)",
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above)",
    r"(r[ée]v[eè]le|affiche|montre)[rz]?\s+(ton|le|votre)\s+(prompt|message)\s+syst[eè]me",
    r"system\s+prompt",
    r"tu\s+es\s+(maintenant|d[ée]sormais|dor[ée]navant)\s+",
    r"you\s+are\s+now\s+",
    r"mode\s+(d[ée]veloppeur|developer)",
    r"jailbreak",
    r"user_role\s*[:=]",
)

# --- orchestrateur ------------------------------------------------------------
ROUTES = ("direct", "recherche", "refus")
MAX_SUB_QUESTIONS = 4
ORCHESTRATOR_JSON_RETRIES = 1  # « un nouvel essai »

# --- chercheur / verificateur (sous-graphe recherche) ------------------------
MAX_RESEARCH_ATTEMPTS = 3       # N : tentatives max par sous-question
MAX_QUERIES_PER_ATTEMPT = 3     # 1 à 3 requêtes par tentative
MAX_TOOL_CALLS_PER_ATTEMPT = 6  # limite d'appels à l'outil par tentative
TOOL_TIMEOUT_SECONDS = 30.0
TOP_K = 5
SUPPORTED_RATIO_MIN = 0.8       # valid = supported_ratio ≥ 0,8 …
RELEVANCE_MIN = 0.6             # … et relevance ≥ 0,6

# --- generateur_final / controle_citations ----------------------------------
MAX_GENERATION_ATTEMPTS = 2     # G : première génération + 1 régénération

# --- budget global par requête ------------------------------------------------
MAX_LLM_CALLS = 40
MAX_LATENCY_SECONDS = 600.0     # les modèles gratuits répondent en 5 à 30 s par appel

# --- messages utilisateur -----------------------------------------------------
FALLBACK_MESSAGE = "Je n'ai pas trouvé d'information disponible pour votre profil dans le corpus."
WARNING_MESSAGE = "réponse générée, vérifiez les sources"
NOT_FOUND_PREFIX = "Ce que je n'ai pas trouvé : "

# Causes de repli (écrites dans la trace, jamais montrées à l'utilisateur)
CAUSE_INPUT_REFUSED = "entree_refusee"
CAUSE_OUT_OF_SCOPE = "hors_perimetre"
CAUSE_NO_VALID_SUB = "aucune_sous_question_valide"
CAUSE_CITATIONS = "echec_citations"
CAUSE_BUDGET = "budget_depasse"
CAUSE_MISSING_NODE = "noeud_indisponible"
CAUSE_LLM = "llm_indisponible"
