# Guide de contribution

Merci de contribuer à **Agentic RAG — Legal Assistant (composant francophone)**.

## Principes

- Le périmètre est le **droit marocain rédigé en français**.
- Aucune affirmation technique ne doit être documentée sans être confirmée par le code ou la documentation.
- Ne jamais présenter une fonctionnalité planifiée comme fonctionnelle.

## Stratégie de branches

| Branche | Règle |
|---|---|
| `production` | Protégée. Modifications via Pull Request |
| `<Prénom/Nom>` | Branche personnelle de chaque contributeur |

## Processus

1. Mettre à jour sa branche personnelle depuis `production`.
2. Travailler sur sa branche, avec des commits atomiques.
3. Ouvrir une Pull Request vers `production` en renseignant le modèle fourni.
4. Attendre la revue du responsable du dépôt (voir `CODEOWNERS`).
5. Corriger les retours, puis fusionner après approbation.

## Messages de commit

Format recommandé : `type: description courte`

Types : `docs`, `feat`, `fix`, `refactor`, `test`, `chore`, `ci`.

## Sécurité et données

- Ne jamais committer de clés d'API, de secrets ni de fichiers `.env`.
- Ne committer que des documents dont la redistribution est autorisée.