"""GraphRAG léger (F-RET-06) : entités et relations extraites du corpus.

Idée : en plus de la similarité texte (vector/BM25), on exploite la STRUCTURE des textes
juridiques. Un graphe relie les chunks aux entités qu'ils citent (dahirs, lois, décrets,
arrêtés, articles, ministères/directions, sigles) et les entités entre elles
(modifie / complète / abroge / remplace / co-occurrence).

À la requête : entités de la question -> voisins dans le graphe -> chunks qui les citent,
classés par score (poids de la relation x rareté de l'entité). Ce classement est ensuite
fusionné par RRF avec les autres (voir advanced_search.py).

Sécurité (F-SEC-04) : le graphe est construit PAR (corpus, rôle) uniquement avec les chunks
que le rôle a le droit de voir. Un lecteur ne peut donc ni obtenir de chunk interdit, ni
"deviner" une entité par ses voisins.

Extraction : règles (regex, rapide, déterministe, gratuit) par défaut ; extracteur LLM
Ollama optionnel (`LLMExtractor`) pour des entités/relations plus riches.
"""
from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .advanced_config import (
    GRAPH_NEIGHBOR_LIMIT,
    GRAPH_WEIGHT_COOCCUR,
    GRAPH_WEIGHT_RELATION,
    GRAPH_WEIGHT_SEED,
)
from .llm import LLM, LLMUnavailable
from .loader import IndexedChunk
from .models import SearchResult
from .vector_store import check_role

log = logging.getLogger(__name__)

# Types d'entités
TEXTE, ARTICLE, ORG, SIGLE = "TEXTE", "ARTICLE", "ORG", "SIGLE"


def fold(text: str) -> str:
    """minuscules, sans accents, espaces normalisés (apostrophes -> espace)."""
    text = unicodedata.normalize("NFKD", text.replace("’", "'"))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    return re.sub(r"[\s']+", " ", text).strip()


@dataclass(frozen=True)
class Mention:
    type: str
    key: str     # forme normalisée (identifiant du noeud)
    label: str   # forme lisible
    start: int = 0
    end: int = 0


@dataclass(frozen=True)
class Relation:
    source: str
    relation: str
    target: str


class Extractor(Protocol):
    def extract(self, text: str) -> tuple[list[Mention], list[Relation]]: ...


# --------------------------------------------------------------------------------------
# Extraction par règles
# --------------------------------------------------------------------------------------
_KIND = r"dahir(?:\s+portant\s+loi)?|loi|d[ée]cret|arr[êe]t[ée](?:\s+(?:viziriel|r[ée]sidentiel|conjoint))?"
_NUM = r"\d+(?:[-/.]\d+)*"
# "dahir n° 1-58-376", "loi n° 16-98", "dahir portant loi n° 1-93-147"
RE_TEXT_NUM = re.compile(rf"\b({_KIND})\s+n[°o*º]?\s*({_NUM})", re.I)
# "arrêté du ministre de la santé n° 1318-04"
RE_ARRETE_NUM = re.compile(r"\b(arr[êe]t[ée])\b[^.\n;]{0,120}?\bn[°o*º]\s*(\d+-\d{2,4})", re.I)
# "dahir du 3 août 1955", "arrêté viziriel du 10 août 1955"
RE_TEXT_DATE = re.compile(
    rf"\b({_KIND})\s+du\s+(\d{{1,2}}(?:er)?\s+[a-zéûôàè]{{3,12}}\s+\d{{4}})", re.I
)
RE_ARTICLE = re.compile(r"\b(?:article|art)\s*\.?\s*(premier|unique|\d{1,3}(?:\.\d{1,2})?(?:\s*(?:bis|ter))?)\b", re.I)
RE_ROMAN = re.compile(r"^[IVXLCDM]+$")
RE_TOKEN = re.compile(r"[\w-]+", re.U)  # l'énergie -> "l" + "énergie"
RE_SIGLE = re.compile(r"\b[A-Z]{3,6}\b")

ORG_HEADS = {
    "ministere", "ministre", "direction", "office", "universite", "faculte",
    "conseil", "commission", "tribunal", "banque", "centre", "cour",
}
ORG_CONNECT = {"de", "du", "des", "d", "la", "l", "et"}
ORG_STOP = {
    "est", "sont", "sera", "seront", "charge", "chargee", "charges", "presente", "present",
    "qui", "que", "dont", "pour", "par", "sur", "dans", "ont", "peut", "doit", "doivent",
    "fixant", "portant", "modifiant", "completant", "abrogeant", "relatif", "relative",
    "autorisant", "approuvant", "ayant", "date", "vu", "arrete", "decision", "decret", "dahir",
    "fait", "sera", "article", "art", "delegue", "auprès", "aupres",
}
MAX_ORG_TOKENS = 10

_VERB_REL = [
    ("modifi", "modifie"), ("compl", "complete"), ("abrog", "abroge"),
    ("rempla", "remplace"), ("proroge", "proroge"),
]
RE_ACTIVE = re.compile(r"\b(modifiant|compl[ée]tant|abrogeant|rempla[cç]ant|prorogeant)\b", re.I)
RE_PASSIVE = re.compile(r"\b(modifi[ée]e?s?|compl[ée]t[ée]e?s?|abrog[ée]e?s?|remplac[ée]e?s?)\s+par\b", re.I)


def _norm_article(raw: str) -> str:
    raw = fold(raw)
    return "1" if raw == "premier" else raw


def _text_key(kind: str, ident: str) -> str:
    kind = re.sub(r"\s+", " ", kind)
    return fold(f"{kind} {ident}")


def _contiguous(text: str, prev_end: int, next_start: int) -> bool:
    """Deux mots appartiennent à la même expression s'ils ne sont séparés que par des blancs."""
    gap = text[prev_end:next_start]
    return gap.replace("'", "").replace("’", "").strip() == "" and gap.count("\n") < 2


def _org_mentions(text: str) -> list[Mention]:
    tokens = [(m.group(0), m.start(), m.end()) for m in RE_TOKEN.finditer(text)]
    found: list[Mention] = []
    i = 0
    while i < len(tokens):
        if fold(tokens[i][0]) not in ORG_HEADS:
            i += 1
            continue
        j, content = i + 1, 0
        while j < len(tokens) and j - i <= MAX_ORG_TOKENS and _contiguous(text, tokens[j - 1][2], tokens[j][1]):
            word = fold(tokens[j][0]).strip("-")
            if word in ORG_CONNECT:
                j += 1
            elif word.isalpha() and len(word) > 2 and word not in ORG_STOP and not (len(word) > 4 and word.endswith("er")):
                content += 1
                j += 1
            else:
                break
        while j > i + 1 and fold(tokens[j - 1][0]) in ORG_CONNECT:  # pas de mot de liaison final
            j -= 1
        if content >= 1 and j > i + 1:
            label = re.sub(r"\s+", " ", text[tokens[i][1] : tokens[j - 1][2]])
            found.append(Mention(ORG, fold(label), label, tokens[i][1], tokens[j - 1][2]))
            i = j
        else:
            i += 1
    return found


class RuleExtractor:
    """Entités et relations par expressions régulières (adapté aux textes juridiques FR/MA)."""

    def __init__(self, known_words: set[str] | None = None):
        # mots vus en minuscules/capitalisés dans le corpus : un MOT COMMUN EN MAJUSCULES
        # (DIRECTION, ARTICLE...) n'est pas un sigle. CNPN, LST, FST n'y figurent jamais.
        self.known_words = known_words

    def extract(self, text: str) -> tuple[list[Mention], list[Relation]]:
        mentions: list[Mention] = []
        for rx in (RE_TEXT_NUM, RE_TEXT_DATE):
            for m in rx.finditer(text):
                mentions.append(Mention(TEXTE, _text_key(m.group(1), m.group(2)),
                                        m.group(0), m.start(), m.end()))
        for m in RE_ARRETE_NUM.finditer(text):
            mentions.append(Mention(TEXTE, _text_key("arrete", m.group(2)), m.group(0), m.start(), m.end()))
        for m in RE_ARTICLE.finditer(text):
            mentions.append(Mention(ARTICLE, f"article {_norm_article(m.group(1))}",
                                    m.group(0), m.start(), m.end()))
        mentions += _org_mentions(text)
        for m in RE_SIGLE.finditer(text):
            word = m.group(0)
            if RE_ROMAN.match(word):  # III, VII... : numéros de section, pas des sigles
                continue
            if self.known_words is not None and word.lower() in self.known_words:
                continue
            mentions.append(Mention(SIGLE, word.lower(), word, m.start(), m.end()))

        # "arrêté du 7 septembre" peut être vu par RE_TEXT_NUM puis RE_TEXT_DATE : on déduplique
        unique: dict[tuple[str, int], Mention] = {}
        for mention in mentions:
            unique.setdefault((mention.key, mention.start), mention)
        mentions = sorted(unique.values(), key=lambda x: x.start)
        return mentions, self._relations(text, mentions)

    @staticmethod
    def _relations(text: str, mentions: list[Mention]) -> list[Relation]:
        texts = [m for m in mentions if m.type == TEXTE]
        relations: list[Relation] = []
        for first, second in zip(texts, texts[1:]):
            if first.key == second.key or second.start - first.end > 200:
                continue
            between = text[first.end : second.start]
            active, passive = RE_ACTIVE.search(between), RE_PASSIVE.search(between)
            if not active and not passive:
                # "portant modification de" juste avant `first` : le sujet est alors le titre du texte
                continue
            verb = (active or passive).group(1).lower()
            rel = next((r for stem, r in _VERB_REL if verb.startswith(stem)), "lie")
            if active:
                relations.append(Relation(first.key, rel, second.key))
            else:  # "X modifié par Y"  =>  Y modifie X
                relations.append(Relation(second.key, rel, first.key))
        return relations


class LLMExtractor:
    """Extraction par LLM local (plus riche, mais lente : ~1 appel par chunk, mis en cache)."""

    SYSTEM = "Tu extrais des entités et relations de textes juridiques marocains. Réponds en JSON."

    def __init__(self, llm: LLM, fallback: Extractor | None = None):
        self.llm = llm
        self.fallback = fallback or RuleExtractor()

    def extract(self, text: str) -> tuple[list[Mention], list[Relation]]:
        mentions, relations = self.fallback.extract(text)  # les règles restent la base
        prompt = (
            "Texte :\n" + text[:2500] + "\n\n"
            "Extrais les entités juridiques (types: TEXTE, ORG, LIEU, PERSONNE, NOTION) et les "
            "relations entre elles (ex: modifie, abroge, complète, relève_de, applique, "
            "responsable_de).\n"
            'JSON : {"entities":[{"name":"...","type":"..."}],'
            '"relations":[{"source":"...","relation":"...","target":"..."}]}'
        )
        try:
            data = json.loads(self.llm.generate(prompt, system=self.SYSTEM, json_mode=True))
        except (LLMUnavailable, json.JSONDecodeError) as exc:
            log.warning("extraction LLM ignorée: %s", exc)
            return mentions, relations
        have = {m.key for m in mentions}
        for ent in data.get("entities", []) if isinstance(data, dict) else []:
            name = str(ent.get("name", "")).strip()
            if len(name) > 2 and fold(name) not in have:
                mentions.append(Mention(str(ent.get("type", "NOTION")).upper(), fold(name), name))
                have.add(fold(name))
        for rel in data.get("relations", []) if isinstance(data, dict) else []:
            src, dst = fold(str(rel.get("source", ""))), fold(str(rel.get("target", "")))
            if src and dst and src != dst:
                relations.append(Relation(src, fold(str(rel.get("relation", "lie"))).replace(" ", "_"), dst))
                for key, raw in ((src, rel.get("source")), (dst, rel.get("target"))):
                    if key not in have:
                        mentions.append(Mention("NOTION", key, str(raw)))
                        have.add(key)
        return mentions, relations


def corpus_vocabulary(chunks: list[IndexedChunk]) -> set[str]:
    """Mots communs du corpus (vus autrement qu'en MAJUSCULES) : sert à rejeter les faux sigles."""
    words: set[str] = set()
    for chunk in chunks:
        for tok in re.findall(r"[^\W\d_]{3,}", chunk.text):
            if not tok.isupper():
                words.add(fold(tok))
    return words


# --------------------------------------------------------------------------------------
# Le graphe
# --------------------------------------------------------------------------------------
class LegalGraph:
    def __init__(self) -> None:
        self.chunks: dict[str, IndexedChunk] = {}
        self.labels: dict[str, str] = {}
        self.types: dict[str, str] = {}
        self.entity_chunks: dict[str, set[str]] = defaultdict(set)
        self.chunk_entities: dict[str, set[str]] = defaultdict(set)
        self.relations: Counter[tuple[str, str, str]] = Counter()
        self.cooccur: dict[str, Counter[str]] = defaultdict(Counter)
        self.extractor: Extractor | None = None

    # -- construction ----------------------------------------------------------------
    @classmethod
    def build(cls, chunks: list[IndexedChunk], extractor: Extractor | None = None) -> "LegalGraph":
        graph = cls()
        graph.extractor = extractor or RuleExtractor(corpus_vocabulary(chunks))
        for chunk in chunks:
            graph._add_chunk(chunk)
        graph._prune_rare_sigles()
        return graph

    def _add_chunk(self, chunk: IndexedChunk) -> None:
        assert self.extractor is not None
        self.chunks[chunk.chunk_id] = chunk
        mentions, relations = self.extractor.extract(chunk.text)
        if chunk.article_number:  # métadonnée d'ingestion : ce chunk EST cet article
            digits = re.search(r"\d+", str(chunk.article_number))
            if digits:
                mentions.append(Mention(ARTICLE, f"article {digits.group(0)}", f"Article {digits.group(0)}"))
        keys: set[str] = set()
        for m in mentions:
            self.labels.setdefault(m.key, m.label)
            self.types.setdefault(m.key, m.type)
            keys.add(m.key)
        for key in keys:
            self.entity_chunks[key].add(chunk.chunk_id)
            self.chunk_entities[chunk.chunk_id].add(key)
        for rel in relations:
            for key in (rel.source, rel.target):
                self.types.setdefault(key, TEXTE)
                self.labels.setdefault(key, key)
                self.entity_chunks[key].add(chunk.chunk_id)
                self.chunk_entities[chunk.chunk_id].add(key)
            self.relations[(rel.source, rel.relation, rel.target)] += 1
        linkable = sorted(k for k in keys if self.types[k] != ARTICLE)  # articles = trop génériques
        for i, a in enumerate(linkable):
            for b in linkable[i + 1 :]:
                self.cooccur[a][b] += 1
                self.cooccur[b][a] += 1

    def _prune_rare_sigles(self) -> None:
        """Un sigle vu dans un seul chunk est presque toujours du bruit OCR."""
        for key in [k for k, t in self.types.items() if t == SIGLE and len(self.entity_chunks[k]) < 2]:
            self._drop(key)

    def _drop(self, key: str) -> None:
        for cid in self.entity_chunks.pop(key, set()):
            self.chunk_entities[cid].discard(key)
        self.cooccur.pop(key, None)
        for other in self.cooccur.values():
            other.pop(key, None)
        self.labels.pop(key, None)
        self.types.pop(key, None)

    # -- requête ---------------------------------------------------------------------
    def idf(self, key: str) -> float:
        n, df = max(len(self.chunks), 1), max(len(self.entity_chunks.get(key, ())), 1)
        return math.log(1.0 + n / df)

    def find_seeds(self, query: str) -> set[str]:
        """Entités de la question qui existent dans le graphe."""
        extractor = self.extractor or RuleExtractor()
        seeds = {m.key for m in extractor.extract(query)[0] if m.key in self.entity_chunks}
        folded = fold(query)
        tokens = re.findall(r"[\w-]+", folded)
        for key, typ in self.types.items():
            if key not in self.entity_chunks or key in seeds:
                continue
            if typ == SIGLE and key in tokens:
                seeds.add(key)
            elif typ == ORG:
                # le nom complet est dans la question, ou la question en cite le début (>= 2 mots)
                if f" {key} " in f" {folded} ":
                    seeds.add(key)
                else:
                    words = key.split()
                    for n in range(min(len(words), 6), 1, -1):
                        if " ".join(words[:n]) in folded and len(words[:n]) >= 2 and words[1] not in ORG_CONNECT:
                            seeds.add(key)
                            break
        return seeds

    def expand(self, seeds: set[str], neighbor_limit: int = GRAPH_NEIGHBOR_LIMIT) -> dict[str, float]:
        """Poids de chaque entité : graines > voisins par relation typée > co-occurrence."""
        weights = {s: GRAPH_WEIGHT_SEED for s in seeds}
        for (src, _rel, dst), _count in self.relations.items():
            if src in seeds:
                weights[dst] = max(weights.get(dst, 0.0), GRAPH_WEIGHT_RELATION)
            if dst in seeds:
                weights[src] = max(weights.get(src, 0.0), GRAPH_WEIGHT_RELATION)
        hub_limit = 0.2 * len(self.chunks) if len(self.chunks) > 50 else float("inf")
        for seed in seeds:
            for neighbor, _count in self.cooccur.get(seed, Counter()).most_common(neighbor_limit):
                if len(self.entity_chunks.get(neighbor, ())) <= hub_limit:
                    weights[neighbor] = max(weights.get(neighbor, 0.0), GRAPH_WEIGHT_COOCCUR)
        return weights

    def search(
        self,
        query: str,
        k: int,
        accept: Callable[[IndexedChunk], bool] | None = None,
    ) -> list[tuple[IndexedChunk, float]]:
        seeds = self.find_seeds(query)
        if not seeds:
            return []
        scores: dict[str, float] = defaultdict(float)
        for key, weight in self.expand(seeds).items():
            for cid in self.entity_chunks.get(key, ()):
                if accept is None or accept(self.chunks[cid]):
                    scores[cid] += weight * self.idf(key)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [(self.chunks[cid], score) for cid, score in ranked]

    # -- inspection / export ------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        by_type = Counter(self.types.values())
        top = sorted(self.entity_chunks, key=lambda e: len(self.entity_chunks[e]), reverse=True)[:10]
        return {
            "chunks": len(self.chunks),
            "entities": len(self.types),
            "entities_by_type": dict(by_type),
            "typed_relations": len(self.relations),
            "cooccurrence_edges": sum(len(v) for v in self.cooccur.values()) // 2,
            "chunks_without_entity": sum(1 for c in self.chunks if not self.chunk_entities.get(c)),
            "top_entities": [(self.labels[e], len(self.entity_chunks[e])) for e in top],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [
                {"id": k, "label": self.labels[k], "type": self.types[k], "chunks": len(self.entity_chunks[k])}
                for k in sorted(self.types)
            ],
            "relations": [
                {"source": s, "relation": r, "target": t, "count": c}
                for (s, r, t), c in sorted(self.relations.items())
            ],
            "cooccurrence": [
                {"a": a, "b": b, "count": c}
                for a, nbrs in sorted(self.cooccur.items())
                for b, c in sorted(nbrs.items()) if a < b and c >= 2
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")


def to_result(chunk: IndexedChunk, score: float, method: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk.chunk_id, text=chunk.text, score=score, method=method, corpus=chunk.corpus,
        title=chunk.title, source_url=chunk.source_url, page_start=chunk.page_start,
        page_end=chunk.page_end, section=chunk.section, article_number=chunk.article_number,
        document_type=chunk.document_type, publication_year=chunk.publication_year,
    )


class GraphStore:
    """Un graphe par (corpus, rôle), construit à la demande — même logique que Bm25Store."""

    def __init__(self, corpora: dict[str, list[IndexedChunk]], extractor_factory: Callable[[list[IndexedChunk]], Extractor] | None = None):
        self.corpora = corpora
        self.extractor_factory = extractor_factory
        self._graphs: dict[tuple[str, str], LegalGraph] = {}

    def graph(self, corpus: str, role: str | None) -> LegalGraph:
        role = check_role(role)  # F-SEC-04 : pas de rôle valide => refus
        if corpus not in self.corpora:
            raise ValueError(f"Unknown corpus '{corpus}'. Choose from {list(self.corpora)}")
        key = (corpus, role)
        if key not in self._graphs:
            allowed = [c for c in self.corpora[corpus] if role in c.allowed_roles]  # filtre AVANT
            extractor = self.extractor_factory(allowed) if self.extractor_factory else None
            self._graphs[key] = LegalGraph.build(allowed, extractor)
        return self._graphs[key]

    def search(
        self, query: str, corpus: str, role: str | None, k: int, filters: dict | None = None
    ) -> list[SearchResult]:
        graph = self.graph(corpus, role)
        accept = None
        if filters:
            accept = lambda c: all(getattr(c, f, None) == v for f, v in filters.items())  # noqa: E731
        return [to_result(c, s, "graph") for c, s in graph.search(query, k, accept)]
