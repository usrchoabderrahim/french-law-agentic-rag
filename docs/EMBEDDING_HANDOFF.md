# Architecture Embeddings – MA Regulatory Assistant

## Objectif

Préparer les chunks dans VS Code (CPU), calculer les embeddings denses avec
`BAAI/bge-m3` dans Google Colab (GPU si disponible), puis livrer au groupe
Retrieval un corpus cohérent avec les identifiants et métadonnées.

## Architecture

1. VS Code : collecte, OCR français, nettoyage, métadonnées et chunking.
2. Export : `data/processed/*.json` vers `data/colab_input/all_chunks.jsonl`.
3. Google Drive : stockage du JSONL d'entrée et des checkpoints.
4. Google Colab : BAAI/bge-m3, embeddings denses normalisés.
5. Livraison : `embeddings.npy`, `embedding_ids.jsonl` et manifeste.

## A. Dans VS Code

Copier `scripts/export_chunks_for_colab.py` dans le dossier `scripts/` du projet.
Depuis la racine du dépôt :

```powershell
python .\scripts\export_chunks_for_colab.py --processed .\data\processed --output .\data\colab_input\all_chunks.jsonl
```

Vérifier les deux fichiers créés :

- `data/colab_input/all_chunks.jsonl`
- `data/colab_input/chunk_export_manifest.json`

Copier le dossier `data/colab_input/` dans Google Drive sous :

`MyDrive/MA_Regulatory_Assistant/colab_input/`

## B. Dans Google Colab

1. Ouvrir `notebooks/generate_embeddings_colab.ipynb`.
2. Dans Colab, choisir **Runtime > Change runtime type > T4 GPU** si T4 est proposé.
3. Exécuter les cellules dans l'ordre.
4. Le notebook vérifie qu'un GPU CUDA est disponible. S'il n'y en a pas, il s'arrête au lieu de lancer silencieusement un long calcul CPU.
5. Le notebook sauvegarde des shards par lots dans Drive. Après une interruption, relancer les cellules : les lots déjà validés sont ignorés.

Le GPU proposé par Colab n'est pas garanti. Si T4 n'est pas proposé, le runtime peut proposer un autre GPU ou aucun GPU selon la disponibilité du service.

## C. Fichiers de livraison

Dans `MyDrive/MA_Regulatory_Assistant/embeddings/bge_m3_v1/` :

- `embeddings.npy` : matrice float32 de forme `(nombre_de_chunks, 1024)`.
- `embedding_ids.jsonl` : correspondance ligne par ligne entre le vecteur et `chunk_id`.
- `embedding_manifest.json` : modèle, dimension, normalisation, hash d'entrée, GPU et statistiques.
- `checkpoints/` : lots sauvegardés pour reprise après interruption.

Le groupe Retrieval doit joindre la ligne `i` de `embeddings.npy` à la ligne `i` de
`embedding_ids.jsonl`, puis retrouver le texte et les métadonnées via `chunk_id`
dans `all_chunks.jsonl`.

## Paramètres de départ

- Modèle : `BAAI/bge-m3`
- Mode : embeddings denses uniquement
- Dimension : 1024
- Normalisation : L2
- Similarité attendue : cosine
- Longueur maximale de tokenisation dans le notebook : 2048 tokens
- Taille de lot de sauvegarde : 256 chunks
- Batch GPU initial : 8, réduction automatique si mémoire GPU insuffisante

La longueur maximale est volontairement limitée à 2048 pour la stabilité du premier
traitement. Le notebook signale les chunks qui dépassent cette limite et refuse de
les tronquer silencieusement. Il faudra alors revoir le chunking ou la limite.

## Contrôles avant livraison

- nombre de chunks exportés = nombre de lignes dans `embedding_ids.jsonl`;
- nombre de vecteurs = nombre de chunks;
- dimension de chaque vecteur = 1024;
- tous les `chunk_id` sont uniques;
- embeddings normalisés;
- SHA-256 du JSONL inscrit dans le manifeste;
- aucun document arabe ne doit être inclus dans le corpus français;
- les chunks OCR de faible confiance doivent être signalés dans les rapports qualité.

## Important

Ce pack prépare le transfert d'embeddings. Il ne réalise pas le travail du groupe
Retrieval : insertion Qdrant, BM25, recherche hybride et reranking restent leur
responsabilité.
