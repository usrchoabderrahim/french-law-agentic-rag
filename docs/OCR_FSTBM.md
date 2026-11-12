# Passation du pipeline OCR et du chunking FSTBM

Cette contribution va de l’import des documents français jusqu’à l’export des chunks avec leur provenance. Le corpus de démonstration contient deux PDF et deux pages HTML de la FST de Béni Mellal. Les trois Bulletins officiels servent à mesurer le traitement de scans historiques. Les règlements de l’administration seront intégrés après la livraison de l’assistant.

## Fichiers et dossiers à reprendre

| Emplacement | Rôle |
| --- | --- |
| `src/ingestion/loaders.py` | Extraction PDF, OCR page par page, lecture du contenu principal des pages HTML et diagnostics OCR. |
| `src/ingestion/chunking.py` | Découpage par section, numéros d’article complets, plages de pages et séparation du sommaire. |
| `src/ingestion/pipeline.py` | Orchestration de l’ingestion, provenance, reprise des traitements et journal des documents incomplets. |
| `scripts/prepare_fstbm_corpus.py` | Préparation des sept sources publiques et production des deux ZIP. |
| `scripts/ingest_local.py` | Import d’un dossier de nouveaux documents, notamment les règlements fournis par la faculté. |
| `scripts/export_chunks_for_colab.py` | Export de `all_chunks.jsonl` et de `chunk_export_manifest.json`. |
| `scripts/benchmark_ocr_preprocessing.py`, `scripts/benchmark_ocr_targeted.py` | Comparaison des réglages OCR sur les scans du Bulletin officiel de 1955. |
| `tests/test_loaders.py`, `tests/test_chunking.py`, `tests/test_pipeline.py` | Vérification de la détection OCR, du découpage et de la reprise après un échec. |
| `data/raw/fstbm_pilot/` | Sources locales téléchargées dans `fstbm/` et `archives_ocr/`. |
| `data/processed/fstbm_corpus/` | ZIP, documents traités et manifestes d’ingestion par lot. |
| `notebooks/generate_embeddings_colab.ipynb` | Génération des vecteurs BGE-M3 sur GPU. [Passation embeddings](EMBEDDING_HANDOFF.md) |

## Résultats vérifiés

- **Démo FSTBM** · 44 chunks issus de quatre documents. Les 11 pages PDF sont couvertes et les pages HTML gardent leur URL officielle.
- **Benchmark OCR** · 981 chunks issus des 119 pages scannées des trois Bulletins officiels. Chaque page a fourni du texte.
- **Détection d’une couche texte partielle** · sur une page FSTBM transformée en scan de test, la nouvelle détection déclenche l’OCR. Le volume de texte extrait passe de 22 à 352 mots pour une référence de 362 mots. Le taux d’erreurs de mots passe de 98,07 % à 14,36 %.
- **Tests automatisés** · 24 tests passent. Les deux ZIP ont été vérifiés avec leur empreinte SHA-256, leurs identifiants uniques et leurs références de page.

Les valeurs des tableaux scannés demandent une vérification sur le PDF. Par exemple, page 45 du bulletin de 1955, `54.270` est lu `54.570`. Certaines références d’acte des bulletins historiques demandent également une revue. Pour la démo, utiliser le lot FSTBM et vérifier qu’une réponse correspond au passage cité.

## Installation

Depuis la racine du dépôt, dans l’environnement Python du projet, installer les dépendances OCR. Installer Tesseract et le modèle de langue `fra`, puis vérifier la présence de `fra` dans la liste.

```powershell
python -m pip install -e '.[ocr]'
tesseract --list-langs
```

Sur Windows, le chargeur reconnaît également l’installation sous `C:\Program Files\Tesseract-OCR`. Pour utiliser un autre emplacement, définir `TESSERACT_CMD` avec le chemin de l’exécutable.

## Sources publiques

Enregistrer les quatre fichiers FSTBM dans `data/raw/fstbm_pilot/fstbm` et les trois bulletins dans `data/raw/fstbm_pilot/archives_ocr`.

| Lot | Nom du fichier | Source |
| --- | --- | --- |
| FSTBM | `normes_complementaires_lst.pdf` | [Normes complémentaires de Licence](https://fstbm.ac.ma/web/content/1489951/Normes_compl%C3%A9mentaires_CNPN-LST.pdf) |
| FSTBM | `charte_these.pdf` | [Charte de thèse](https://fstbm.ac.ma/web/content/1417391?download=true) |
| FSTBM | `futurs_etudiants.html` | [Futurs étudiants](https://fstbm.ac.ma/futurs-etudiants) |
| FSTBM | `guide_e_scolarite.html` | [Guide E-Scolarité](https://fstbm.ac.ma/blog/blog-de-l-espace-etudiant-1/post/comment-s-inscrire-sur-la-plateforme-e-scolarite-1) |
| Benchmark OCR | `bo_2240_1955.pdf` | [Bulletin officiel n° 2240](https://www.sgg.gov.ma/BO/bo_fr/1955/bo_2240_fr.pdf) |
| Benchmark OCR | `bo_4478_1997.pdf` | [Bulletin officiel n° 4478](https://www.sgg.gov.ma/BO/bo_fr/1997/bo_4478_fr.pdf) |
| Benchmark OCR | `bo_5236_2004.pdf` | [Bulletin officiel n° 5236](https://www.sgg.gov.ma/BO/bo_fr/2004/bo_5236_fr.pdf) |

## Exécution et contrôles

```powershell
python scripts/prepare_fstbm_corpus.py --group all
```

Les ZIP sont écrits dans `data/processed/fstbm_corpus`. `--group fstbm` prépare seulement le lot de démonstration. `--group archives_ocr` prépare le benchmark sur les PDF originaux. Utiliser `--reprocess` pour relancer après un changement du code ou du modèle OCR. Les lots de démonstration et de benchmark ont chacun leur manifeste d’ingestion.

Pour exécuter les tests depuis la racine du dépôt, installer les dépendances de développement, puis lancer pytest.

```powershell
python -m pip install -e '.[ocr,dev]'
python -m pytest
```

Le chargeur choisit par page entre la couche texte et l’OCR Tesseract en français. Il utilise la surface occupée par l’image et la présence de texte dans son corps pour détecter un scan dont seul l’en-tête est sélectionnable. Les métadonnées des PDF indiquent le mode choisi, le motif et, en cas d’OCR, la durée. Un échec OCR indique les pages à reprendre et empêche de produire un nouvel export pour le lot concerné.

Pour la démonstration, transmettre `fstbm_chunks_colab.zip` et extraire ses deux fichiers dans `colab_input` pour le notebook d’embeddings. Vérifier quelques réponses à partir des passages récupérés, notamment une référence d’article à plusieurs niveaux et une citation de page. Pour les tableaux, vérifier manuellement l’association de chaque valeur avec sa ligne.

## Passage à l’équipe suivante

1. Publier les corrections et les tests sur la branche personnelle `Sisbane`. Ouvrir ou mettre à jour la pull request avec `production` comme branche cible.
2. Après la revue, le responsable fusionne la pull request. L’équipe suivante récupère alors `production` et poursuit sur sa propre branche de travail.
3. Partager le ZIP FSTBM séparément du code. Après génération, remettre au groupe Retrieval `all_chunks.jsonl`, `embeddings.npy`, `embedding_ids.jsonl` et le manifeste d’embeddings. Le groupe peut relier les passages et les vecteurs par `chunk_id`.

## Règlements transmis après la livraison

Placer les PDF approuvés dans un dossier dédié et leur donner un espace de traitement séparé.

```powershell
python scripts/ingest_local.py --input data/raw/fstbm_reglements --processed data/processed/fstbm_reglements --manifest data/manifests/fstbm_reglements.jsonl --source-name 'FST Béni Mellal'
```

Vérifier le journal et comparer quelques pages scannées au PDF, en particulier les numéros d’article, les dates et les tableaux. Exporter ensuite les chunks du dossier traité pour générer les embeddings des nouveaux documents.
