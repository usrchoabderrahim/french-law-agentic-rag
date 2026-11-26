"""Génère des questions de test à partir des chunks (vérité terrain = le chunk source).

    python -m scripts.generate_eval_set --corpus ocr --n 30

Le LLM local écrit, pour chaque chunk tiré au hasard, une question dont la réponse est dans ce
chunk. Résultat : data/eval/questions_generated.jsonl.  RELISEZ-LE avant de vous en servir :
supprimez les questions floues ou ambiguës, et complétez avec de vraies questions d'utilisateurs.
(Limite connue : d'autres chunks peuvent aussi répondre ; la métrique est alors pessimiste.)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.advanced_config import EVAL_DIR
from src.retrieval.llm import LLMUnavailable, OllamaClient
from src.retrieval.loader import load_corpus

SYSTEM = "Tu prépares des jeux de test pour un moteur de recherche juridique. Réponds en JSON."


def usable(chunk) -> bool:
    """Chunks assez longs et lisibles (on écarte tables des matières, listes de chiffres, bruit OCR)."""
    text = chunk.text
    if len(text) < 250 or chunk.section == "TABLE_OF_CONTENTS" or text.count("....") > 2:
        return False
    letters = sum(c.isalpha() for c in text)
    return letters / len(text) >= 0.65


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, choices=["fstbm", "ocr"])
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--role", default="reader")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "questions_generated.jsonl")
    args = parser.parse_args()

    llm = OllamaClient()
    if not llm.available():
        sys.exit(f"Ollama indisponible ou modèle '{llm.model}' absent (ollama pull {llm.model}).")

    chunks = [c for c in load_corpus(args.corpus) if args.role in c.allowed_roles and usable(c)]
    random.Random(args.seed).shuffle(chunks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("a", encoding="utf-8") as out:
        for chunk in chunks:
            if written >= args.n:
                break
            prompt = (
                f"Passage :\n{chunk.text[:1800]}\n\n"
                "Écris UNE question précise qu'un utilisateur pourrait poser et dont la réponse se "
                "trouve dans ce passage. Ne recopie pas de longues phrases du passage, ne cite pas "
                "'ce passage' ni 'le texte'. La question doit être compréhensible seule.\n"
                'Réponds : {"question": "..."}'
            )
            try:
                question = json.loads(llm.generate(prompt, system=SYSTEM, json_mode=True)).get("question", "")
            except (LLMUnavailable, json.JSONDecodeError, AttributeError):
                continue
            question = re.sub(r"\s+", " ", str(question)).strip()
            if len(question) < 15:
                continue
            written += 1
            row = {"id": f"gen_{args.corpus}_{written:03d}", "question": question, "corpus": args.corpus,
                   "role": args.role, "relevant_chunk_ids": [chunk.chunk_id]}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{row['id']}: {question}")
    print(f"\n{written} questions ajoutées à {args.out} — à relire avant évaluation.")


if __name__ == "__main__":
    main()
