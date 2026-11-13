# Génération des embeddings

Ce document décrit le travail réalisé : la génération des embeddings des chunks exportés par le groupe OCR, avec le modèle BAAI/bge-m3 sur Google Colab. Deux lots ont été traités : la démo FSTBM et le benchmark OCR.

## Résultats vérifiés

| Lot | Chunks | Vecteurs | Usage |
| --- | --- | --- | --- |
| Démo FSTBM | 44 | 44 × 1 024 | Démo de l'assistant |
| Benchmark OCR | 981 | 981 × 1 024 | Évaluation séparée |

- **Modèle** · BAAI/bge-m3, dimension 1 024, vecteurs normalisés, similarité cosinus.
- **Matériel** · GPU T4 sur Google Colab. Le calcul du lot OCR a duré environ 0,7 minute.
- **Checkpoints du lot OCR** · 4 checkpoints (256 + 256 + 256 + 213 chunks), soit 981 au total.
- **Manifeste** · `"status": "completed"`, `"dimension": 1024`, `"normalized": true`, avec `count` égal à 44 pour la démo et 981 pour l'OCR.
- **Empreintes SHA-256 des entrées** · `e694a20d...` pour le lot FSTBM et `1f546758...` pour le lot OCR, identiques à celles des ZIP fournis par le groupe OCR.
- **Tailles des fichiers** · `embeddings.npy` fait environ 177 Ko pour la démo et 4 018 304 octets pour l'OCR (981 × 1 024 × 4, plus 128 octets d'en-tête). `all_chunks.jsonl` fait 160 Ko pour la démo et 12 069 959 octets pour l'OCR.

## Ce qui a été fait

1. **Entrées** · les fichiers `all_chunks.jsonl` et le manifeste d'export de chaque ZIP ont été déposés sur Google Drive : dans `colab_input` pour la démo FSTBM et dans `colab_input_ocr` pour le benchmark OCR.
2. **Génération** · les notebooks Colab ont été exécutés avec le GPU T4. Chaque lot produit trois fichiers : `embeddings.npy`, `embedding_ids.jsonl` et `embedding_manifest.json`.
3. **Sorties sur Drive** · la démo est écrite dans `embeddings/bge_m3_v1` et le benchmark OCR dans `embeddings/bge_m3_ocr_v1`. Les deux dossiers sont séparés : le lot OCR n'a pas modifié la démo.
4. **Correction d'un bug d'écriture** · l'écriture des fichiers `.jsonl` utilise maintenant `newline="\n"`, dans les deux notebooks.
5. **Téléchargement et vérification** · les trois fichiers de chaque lot ont été téléchargés depuis Drive dans un dossier à part (`embeddings_fstbm` et `embeddings_ocr`), avec le `all_chunks.jsonl` correspondant. Un script PowerShell a confirmé trois points : le nombre de chunks et d'identifiants (44 puis 981), l'ordre identique (ligne i du `.npy` = ligne i de `embedding_ids.jsonl` = chunk i), et le même fichier d'entrée que dans Colab (empreinte SHA-256 comparée au manifeste). Résultat : `OK : les 44 identifiants correspondent dans le même ordre.` puis `OK : les 981 identifiants correspondent dans le même ordre.`
6. **Publication du code** · les deux notebooks exécutés ont été poussés sur la branche `Jihane` du dépôt `french-law-agentic-rag` (commit `8540cf5`).

## Fichiers produits

| Emplacement | Rôle |
| --- | --- |
| `notebooks/generate_embeddings_colab.ipynb` | Génération des vecteurs de la démo FSTBM (lit `colab_input`, écrit dans `bge_m3_v1`). |
| `notebooks/generate_embeddings_colab_ocr.ipynb` | Génération des vecteurs du benchmark OCR (lit `colab_input_ocr`, écrit dans `bge_m3_ocr_v1`). |
| `embeddings_fstbm/` | Lot démo : `embeddings.npy`, `embedding_ids.jsonl`, `embedding_manifest.json`, `all_chunks.jsonl`. |
| `embeddings_ocr/` | Lot benchmark OCR : mêmes quatre fichiers. |
