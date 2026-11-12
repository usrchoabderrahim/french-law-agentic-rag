from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from ingestion.hashing import stable_chunk_id
from ingestion.schemas import DocumentChunk, LoadedDocument

# ---------------------------------------------------------------------------
# Reconnaissance des structures juridiques
# ---------------------------------------------------------------------------

ARTICLE_LINE_RE = re.compile(
    r"^\s*(?:ARTICLE|ART\.?|ARL\.?|ANT\.?|AUR\.?)\s+"
    r"(?P<number>PREMIER|PREMIÈRE|UNIQUE|1ER|\d{1,3}(?:[A-Z])?(?:[°º])?"
    r"(?:\s+(?:BIS|TER|QUATER))?)"
    r"\s*(?:[.:,;]\s*)?(?:[—–-]\s*)?(?P<rest>.*)$",
    re.IGNORECASE,
)

ACT_HEADING_RE = re.compile(
    r"^\s*(?P<heading>" r"(?:DAHIR|LOI|D[ÉE]CRET|ARR[ÊE]T[ÉE]|ORDONNANCE)" r"\b[^\n]{0,240})$",
    re.IGNORECASE,
)

# A real act title must contain a date, a legal reference number, or a
# descriptive legal qualifier. This rejects procedural formulas such as
# "ARRÊTE :" and OCR fragments such as "arrêté.".
ACT_TITLE_QUALIFIER_RE = re.compile(
    r"\b(?:du|de|des|d|portant|relatif|relative|fixant|modifiant|"
    r"completant|abrogeant|autorisant|approuvant|declarant|rendant|"
    r"instituant|classant|declassant|homologuant|reconnaissant|"
    r"determinant|accordant|concernant|organisant|applicable|"
    r"reglementant|etablissant|creant|ratifiant|visant)\b",
    re.IGNORECASE,
)
ACT_NUMBER_RE = re.compile(
    r"\b(?:n\s*[°ºo]\s*[\w./-]+|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
    re.IGNORECASE,
)
ACT_REFERENCE_RE = re.compile(
    r"\b(?:b|r)\s*[. ]?\s*[o0]\s*[. ]?\s*n\s*(?:[°ºo])?",
    re.IGNORECASE,
)
ACT_TITLE_CONTINUATION_RE = re.compile(
    r"^\s*(?:viziriel|residentiel|du|de|des|d['’]|n\s*[°ºo]|"
    r"\d{1,2}[-/]\d{1,2}|portant|relatif|relative|fixant|modifiant|"
    r"completant|abrogeant|autorisant|approuvant|declarant|rendant|"
    r"instituant|classant|declassant|homologuant)\b",
    re.IGNORECASE,
)

STRUCTURE_HEADING_RE = re.compile(
    r"^\s*(?P<heading>(?:"
    r"(?:SOUS[- ]SECTION|CHAPITRE|TITRE|SECTION|LIVRE|PARTIE)"
    r"\s+(?:PREMIER|PREMIÈRE|DEUXIÈME|DEUXIEME|TROISIÈME|TROISIEME|"
    r"[IVXLCDM]+|\d+)"
    r"|(?:PREMIER|PREMIÈRE|DEUXIÈME|DEUXIEME|TROISIÈME|TROISIEME)"
    r"\s+(?:PARTIE|SECTION|CHAPITRE|TITRE|LIVRE)"
    r")[^\n]*)$",
    re.IGNORECASE,
)

DOCUMENT_SECTION_RE = re.compile(
    r"^\s*(?:"
    r"DIRECTION\s+(?:DE|DES|DU|D['’])\b|"
    r"MINIST[EÈ]RE\s+(?:DE|DES|DU|D['’])\b|"
    r"SERVICE\s+(?:DE|DES|DU|D['’])\b|"
    r"ANNEXE(?:\s|$)|"
    r"SOMMAIRE(?:\s|$)"
    r")",
    re.IGNORECASE,
)

PROCEDURAL_HEADING_RE = re.compile(
    r"^\s*(?:ARR[ÊE]TE\s*[:!.,;]?|LE\s+GRAND\s+VIZIR\b|"
    r"EN\s+CONSEIL\b|VU\s+ET\s+APPROUV[EÉ]\b)\s*$",
    re.IGNORECASE,
)

SIGNATURE_RE = re.compile(
    r"^\s*(?:"
    r"FAIT\s+(?:À|A)?\s*RABAT\b|"
    r"RABAT,?\s+LE\s+\d{1,2}\b|"
    r"VU\s+POUR\s+PROMULGATION\b|"
    r"POUR\s+LE\s+COMMISSAIRE\s+RÉSIDENT[- ]GÉNÉRAL\b|"
    r"LE\s+MINISTRE\s+PLÉNIPOTENTIAIRE\b|"
    r"DÉLÉGUÉ\s+À\s+LA\s+RÉSIDENCE\b|"
    r"F\.\s*(?:DE\s*)?PANAFIEU\b|"
    r"MOHAMED\s+EL\s+MOKRI\b"
    r")",
    re.IGNORECASE,
)

REFERENCES_RE = re.compile(r"^\s*RÉFÉRENCES?\s*:", re.IGNORECASE)

FINANCIAL_LABEL_RE = re.compile(
    r"^(?:"
    r"excedent\s+de\s+recettes|"
    r"prestations?\b|"
    r"produit\s+des\s+prestations|"
    r"restes?\s+a\s+(?:recouvrer|payer|mandater)|"
    r"peages?\b|"
    r"travaux\b|"
    r"vehicules?\s+industriels|"
    r"fonctionnement\b|"
    r"transformation\s+d['’]emplois|"
    r"relevement\s+de\s+credits|"
    r"fournitures\b|"
    r"assurances\b|"
    r"achat\b|"
    r"indemnites?\b|"
    r"total\s+des\s+(?:recettes|depenses)|"
    r"report\s+de\s+credits|"
    r"recettes\b|"
    r"depenses\b|"
    r"materiel\b|"
    r"entretien\b"
    r")"
)

LEGAL_OPENING_RE = re.compile(
    r"^(?:"
    r"le|la|les|l['’]|un|une|des|du|de|d['’]|"
    r"tout|toute|tous|toutes|aucun|aucune|nul|nulle|"
    r"sont|sonl|est|sera|seront|serait|doit|doivent|"
    r"peut|peuvent|ne|si|a|en|par|pour|dans|"
    r"lorsque|lorsqu|il|elle|ils|elles|chaque|"
    r"ce|cet|cette|ces|qui|quand|sous|sur|aux|"
    r"apres|avant|conformement|a\s+titre|a\s+compter|"
    r"charge|chargee|charges|chargees"
    r")\b"
)

AMOUNT_RE = re.compile(r"(?<!\w)(?:\d{1,3}(?:[ .]\d{3}){1,3}|\d{5,})" r"(?:[,.]\d{1,2})?(?!\w)")

LEADER_RE = re.compile(r"(?:\.{2,}|…{2,}|_{3,}|·{3,})")

FRONT_MATTER_MARKERS = (
    "quarante-quatrieme annee",
    "empire cherifien",
    "protectorat de la republique francaise",
    "le bulletin officiel parait",
    "abonnements sont recus",
    "prix des annonces",
    "changement d adresse",
)

TOC_LABEL_RE = re.compile(
    r"\b(?:sommaire|textes generaux|textes particuliers)\b",
    re.IGNORECASE,
)
PRINTED_PAGE_NUMBER_RE = re.compile(r"^\s*\d{3,4}\s*$")


@dataclass(frozen=True)
class HeadingEvent:
    start: int
    heading: str
    kind: str
    heading_type: str | None = None
    article_number: str | None = None


def _fold_text(text: str) -> str:
    """Lowercase and remove accents for robust OCR-oriented comparisons."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower()


def _normalize_article_number(value: str) -> str:
    """Normalize article numbers while preserving PREMIER and 1ER."""
    number = re.sub(r"\s+", " ", value.strip().upper())

    if number in {"PREMIER", "1ER"}:
        return number

    number = re.sub(r"[°º]$", "", number)
    number = re.sub(r"\s+(BIS|TER|QUATER)$", r" \1", number)
    return number


def _normalize_text(text: str) -> str:
    """Normalize line endings without changing the legal wording."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _join_pages(document: LoadedDocument):
    """Join pages while retaining their original positions."""
    parts: list[str] = []
    spans: list[tuple[int, int, int | None, str | None]] = []
    cursor = 0

    for page in document.pages:
        text = _normalize_text(page.text)

        if not text:
            continue

        if parts:
            parts.append("\n\n")
            cursor += 2

        start = cursor
        parts.append(text)
        cursor += len(text)

        spans.append((start, cursor, page.page_number, page.section))

    return "".join(parts), spans


def _page_info(
    start: int,
    end: int,
    spans: list[tuple[int, int, int | None, str | None]],
):
    """Return the source pages intersecting a chunk."""
    overlapping = [span for span in spans if start < span[1] and end > span[0]]

    source_pages = list(dict.fromkeys(span[2] for span in overlapping if span[2] is not None))

    page_section = next(
        (span[3] for span in overlapping if span[3]),
        None,
    )

    if not source_pages:
        return None, None, [], page_section

    return (
        source_pages[0],
        source_pages[-1],
        source_pages,
        page_section,
    )


def _candidate_context(lines: list[str], index: int, rest: str) -> str:
    """Read the article heading and a few following lines for classification."""
    context_parts = [rest.strip()]

    for following_line in lines[index + 1 : index + 4]:
        cleaned = following_line.strip()
        if cleaned:
            context_parts.append(cleaned)

    return " ".join(part for part in context_parts if part)


def _previous_nonempty_line(lines: list[str], index: int) -> str:
    """Return the nearest non-empty line before ``index``."""
    for previous_index in range(index - 1, -1, -1):
        candidate = lines[previous_index].strip()
        if candidate:
            return candidate
    return ""


def _next_nonempty_line(lines: list[str], index: int) -> str:
    """Return the nearest non-empty line after ``index``."""
    for next_index in range(index + 1, len(lines)):
        candidate = lines[next_index].strip()
        if candidate:
            return candidate
    return ""


def _next_nonempty_lines(
    lines: list[str],
    index: int,
    limit: int = 2,
) -> list[str]:
    """Return up to ``limit`` non-empty lines after ``index``."""
    result: list[str] = []
    for next_index in range(index + 1, len(lines)):
        candidate = lines[next_index].strip()
        if candidate:
            result.append(candidate)
            if len(result) >= limit:
                break
    return result


def _is_reference_act_heading(lines: list[str], index: int) -> bool:
    """Reject act-like lines that are citations in a recital/reference list."""
    line = lines[index].strip()
    previous = _previous_nonempty_line(lines, index)
    following = _next_nonempty_line(lines, index)

    folded_line = _fold_text(line)
    folded_previous = _fold_text(previous)

    # Official-gazette references are citations, not new acts.
    if ACT_REFERENCE_RE.search(folded_line):
        return True

    # OCR may wrap "Vu le dahir..." across two physical lines.
    if re.match(
        r"^(?:vu\s+(?:le|la|les|l['’])\b|vu\s*[:;,]?$|"
        r"notamment\b|les textes\b|textes\b|"
        r"qui l['’]ont modifie\b|conformement\b)",
        folded_previous,
    ):
        return True

    # A citation list often continues with another date on the next line.
    if line.rstrip().endswith((",", ";")) and re.match(
        r"^\d{1,2}\s+[a-zà-ÿ-]+\s+\d{4}\b",
        following,
        re.IGNORECASE,
    ):
        return True

    # If a BO/RO reference is adjacent to a comma/semicolon-terminated item,
    # treat the item as part of the reference list.
    nearby = " ".join(lines[max(0, index - 2) : min(len(lines), index + 2)])
    return bool(
        ACT_REFERENCE_RE.search(_fold_text(nearby)) and line.rstrip().endswith((",", ";", "."))
    )


def _has_act_title_qualifier(text: str) -> bool:
    """Check that an act-like heading contains meaningful title information."""
    folded = _fold_text(text)
    return bool(ACT_TITLE_QUALIFIER_RE.search(folded) or ACT_NUMBER_RE.search(folded))


def _is_valid_act_heading(lines: list[str], index: int) -> bool:
    """Validate an act title and reject generic procedural/citation lines."""
    line = lines[index].strip()
    if not ACT_HEADING_RE.match(line):
        return False

    if PROCEDURAL_HEADING_RE.match(line):
        return False

    if _is_reference_act_heading(lines, index):
        return False

    if _has_act_title_qualifier(line):
        return True

    # Some OCR extracts split a title after "Arrêté" onto the next line.
    continuation_lines = _next_nonempty_lines(lines, index, limit=2)
    if continuation_lines and ACT_TITLE_CONTINUATION_RE.match(continuation_lines[0]):
        continuation = " ".join(continuation_lines)
        return _has_act_title_qualifier(continuation)

    return False


def _is_financial_table_row(context: str) -> bool:
    """
    Reject common budget-table rows misread by OCR as legal articles.

    A legal provision usually starts with a grammatical construction
    (Le, Les, Sont, Est, À titre, etc.). Budget rows usually start with
    a label such as "Prestations", "Excédent de recettes" or "Travaux",
    followed by an amount or dot leaders.
    """
    folded = _fold_text(context)
    if not folded:
        return False

    # Preserve grammatical legal clauses, including clauses about budgets.
    if LEGAL_OPENING_RE.match(folded):
        return False

    if FINANCIAL_LABEL_RE.match(folded):
        return True

    has_amount = bool(AMOUNT_RE.search(context))
    has_leader = bool(LEADER_RE.search(context))

    if has_amount and has_leader:
        return True

    first_line = folded.splitlines()[0]
    return bool(has_amount and len(first_line) <= 120)


def _is_caps_heading(line: str) -> bool:
    """
    Detect meaningful administrative headings without treating legal formulas
    ("LE GRAND VIZIR", "ARRÊTE :") as document sections.
    """
    cleaned = re.sub(r"\s+", " ", line).strip(" \t.-:—–")
    if not cleaned or PROCEDURAL_HEADING_RE.match(cleaned):
        return False

    if DOCUMENT_SECTION_RE.match(cleaned):
        return True

    # Retain only clearly labelled all-caps section headings. A broad
    # uppercase heuristic was splitting nearly every preamble/signature line.
    if not 6 <= len(cleaned) <= 110:
        return False

    letters = [char for char in cleaned if char.isalpha()]
    if len(letters) < 5:
        return False

    uppercase_ratio = sum(char.isupper() for char in letters) / len(letters)
    if uppercase_ratio < 0.90:
        return False

    folded = _fold_text(cleaned)
    return bool(
        re.match(
            r"^(?:DIRECTION\b|MINISTERE\b|SERVICE\b|ANNEXE\b|"
            r"SOMMAIRE\b|PREMIERE PARTIE\b|DEUXIEME PARTIE\b|"
            r"TROISIEME PARTIE\b)",
            folded,
        )
    )


def _classify_page_text(page_text: str) -> str:
    """Classify gazette pages while protecting mixed legal pages."""

    lines = [line.strip() for line in page_text.splitlines() if line.strip()]

    if not lines:
        return "LEGAL_CONTENT"

    folded = _fold_text(page_text)

    article_count = sum(bool(ARTICLE_LINE_RE.match(line)) for line in lines)

    act_count = sum(_is_valid_act_heading(lines, index) for index, _line in enumerate(lines))

    leader_count = sum(bool(LEADER_RE.search(line)) for line in lines)

    printed_page_count = sum(bool(PRINTED_PAGE_NUMBER_RE.fullmatch(line)) for line in lines)

    front_matter_hits = sum(marker in folded for marker in FRONT_MATTER_MARKERS)

    if article_count == 0 and act_count == 0 and front_matter_hits >= 2:
        return "FRONT_MATTER"

    # Strong evidence of actual legal provisions.
    legal_markers = (
        "le grand vizir",
        "en conseil restreint",
        "sur la proposition",
        "vu pour promulgation",
        "fait a rabat",
        "le commissaire resident general",
        "le ministre plenipotentiaire",
    )

    legal_marker_hits = sum(marker in folded for marker in legal_markers)

    # Legal content takes priority over TOC heuristics.
    if article_count > 0 or legal_marker_hits >= 2:
        return "LEGAL_CONTENT"

    has_toc_label = bool(TOC_LABEL_RE.search(folded))

    dense_toc_listing = (
        article_count == 0 and act_count >= 3 and (leader_count >= 1 or printed_page_count >= 1)
    )

    if article_count == 0 and (has_toc_label or dense_toc_listing):
        return "TABLE_OF_CONTENTS"

    return "LEGAL_CONTENT"


def _collect_heading_events(
    text: str,
    page_spans: list[tuple[int, int, int | None, str | None]] | None = None,
) -> list[HeadingEvent]:
    """Find validated legal headings and add page-level document boundaries."""
    lines = text.splitlines()
    events: list[HeadingEvent] = []
    page_classes: list[tuple[int, int, str]] = []

    # Add a boundary only when the page class changes. Consecutive TOC pages
    # therefore remain one searchable segment rather than one chunk per page.
    if page_spans:
        previous_page_class: str | None = None
        for page_start, page_end, _page_number, _section in page_spans:
            page_class = _classify_page_text(text[page_start:page_end])
            page_classes.append((page_start, page_end, page_class))
            if page_class != previous_page_class:
                events.append(
                    HeadingEvent(
                        start=page_start,
                        heading=page_class,
                        kind="PAGE_CLASS",
                        heading_type=page_class,
                    )
                )
            previous_page_class = page_class

    offset = 0
    page_class_index = 0
    in_signature_block = False

    for index, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            offset += len(line) + 1
            continue

        while page_class_index < len(page_classes) and offset >= page_classes[page_class_index][1]:
            page_class_index += 1

        current_page_class = "LEGAL_CONTENT"
        if page_class_index < len(page_classes):
            page_start, page_end, page_class = page_classes[page_class_index]
            if page_start <= offset < page_end:
                current_page_class = page_class

        # Cover and contents pages are retained as searchable chunks, but
        # their act-like entries must never create legal act identifiers.
        if current_page_class in {"FRONT_MATTER", "TABLE_OF_CONTENTS"}:
            in_signature_block = False
            offset += len(line) + 1
            continue

        # A genuine act title starts a new legal act and ends a prior
        # signature block.
        if _is_valid_act_heading(lines, index):
            in_signature_block = False
            act_match = ACT_HEADING_RE.match(line)
            assert act_match is not None
            act_heading = act_match.group("heading").strip()
            if not _has_act_title_qualifier(act_heading):
                continuation_lines = _next_nonempty_lines(
                    lines,
                    index,
                    limit=2,
                )
                if continuation_lines and ACT_TITLE_CONTINUATION_RE.match(continuation_lines[0]):
                    act_heading = " ".join([act_heading, *continuation_lines])

            events.append(
                HeadingEvent(
                    start=offset,
                    heading=act_heading,
                    kind="ACT",
                    heading_type="ACT",
                )
            )
            offset += len(line) + 1
            continue

        article_match = ARTICLE_LINE_RE.match(line)
        if article_match:
            number = _normalize_article_number(article_match.group("number"))
            rest = article_match.group("rest").strip()
            rest = re.sub(r"""^[\s"'“”».,;:—–-]+""", "", rest)
            context = _candidate_context(lines, index, rest)

            if _is_financial_table_row(context):
                events.append(
                    HeadingEvent(
                        start=offset,
                        heading=stripped,
                        kind="TABLE_ROW",
                        heading_type="TABLE_ROW",
                    )
                )
            else:
                in_signature_block = False
                events.append(
                    HeadingEvent(
                        start=offset,
                        heading=stripped,
                        kind="ARTICLE",
                        heading_type="ARTICLE",
                        article_number=number,
                    )
                )

            offset += len(line) + 1
            continue

        # Emit one boundary for the complete signature block, not one event
        # for every signature line (date, promulgation, title and name).
        if SIGNATURE_RE.match(line):
            if not in_signature_block:
                events.append(
                    HeadingEvent(
                        start=offset,
                        heading=stripped,
                        kind="SIGNATURE",
                        heading_type="SIGNATURE",
                    )
                )
                in_signature_block = True

            offset += len(line) + 1
            continue

        if in_signature_block:
            # The signature block continues until a new act/article or a
            # clearly identified administrative section is encountered.
            if _is_caps_heading(line):
                in_signature_block = False
            else:
                offset += len(line) + 1
                continue

        structure_match = STRUCTURE_HEADING_RE.match(line)
        if structure_match:
            heading = structure_match.group("heading").strip()
            kind_match = re.match(
                r"(?i)^(SOUS[- ]SECTION|CHAPITRE|TITRE|SECTION|LIVRE|PARTIE)",
                heading,
            )
            if kind_match is None:
                kind_match = re.search(
                    r"(?i)\b(PARTIE|SECTION|CHAPITRE|TITRE|LIVRE)$",
                    heading,
                )
            heading_type = kind_match.group(1).upper() if kind_match else "SECTION"
            events.append(
                HeadingEvent(
                    start=offset,
                    heading=heading,
                    kind="STRUCTURE",
                    heading_type=heading_type,
                )
            )
            offset += len(line) + 1
            continue

        if REFERENCES_RE.match(line):
            events.append(
                HeadingEvent(
                    start=offset,
                    heading=stripped,
                    kind="REFERENCES",
                    heading_type="REFERENCES",
                )
            )
            offset += len(line) + 1
            continue

        if _is_caps_heading(line):
            events.append(
                HeadingEvent(
                    start=offset,
                    heading=stripped,
                    kind="DOCUMENT_SECTION",
                    heading_type="DOCUMENT_SECTION",
                )
            )

        offset += len(line) + 1

    # Remove duplicate events of the same kind at the same offset. Different
    # event kinds at one offset are retained (PAGE_CLASS must precede ACT).
    unique_events: list[HeadingEvent] = []
    seen_events: set[tuple[int, str]] = set()

    for event in sorted(events, key=lambda item: item.start):
        event_key = (event.start, event.kind)
        if event_key not in seen_events:
            unique_events.append(event)
            seen_events.add(event_key)

    return unique_events


def _extract_heading_metadata(
    heading: str | None,
) -> tuple[str | None, str | None]:
    """Compatibility helper: extract heading type and article number."""
    if not heading:
        return None, None

    match = ARTICLE_LINE_RE.match(heading)
    if match:
        return (
            "ARTICLE",
            _normalize_article_number(match.group("number")),
        )

    act_match = ACT_HEADING_RE.match(heading)
    if act_match and _has_act_title_qualifier(heading):
        return "ACT", None

    structure_match = STRUCTURE_HEADING_RE.match(heading)
    if structure_match:
        return "SECTION", None

    return None, None


def _section_ranges(
    text: str,
    document_id: str = "document",
    page_spans: list[tuple[int, int, int | None, str | None]] | None = None,
) -> Iterator[
    tuple[
        str | None,
        int,
        int,
        str | None,
        str | None,
        str | None,
        str | None,
        str | None,
    ]
]:
    """
    Split text at validated legal/document headings.

    Financial table rows and signature blocks reset article metadata.
    Only validated act titles create new act identifiers; procedural formulas
    and cited historical acts do not.
    """
    events = _collect_heading_events(text, page_spans=page_spans)

    if not events:
        yield None, 0, len(text), None, None, None, None, None
        return

    cursor = 0
    current_heading: str | None = None
    current_heading_type: str | None = None
    current_article_number: str | None = None
    current_article_id: str | None = None
    current_act_id: str | None = None
    current_act_title: str | None = None
    act_counter = 0

    for event in events:
        if event.start > cursor:
            yield (
                current_heading,
                cursor,
                event.start,
                current_heading_type,
                current_article_number,
                current_act_id,
                current_act_title,
                current_article_id,
            )

        current_heading = event.heading
        current_heading_type = event.heading_type
        current_article_number = None
        current_article_id = None

        if event.kind == "PAGE_CLASS":
            # Reset only when entering non-legal pages.
            if event.heading_type in {
                "FRONT_MATTER",
                "TABLE_OF_CONTENTS",
            }:
                current_act_id = None
                current_act_title = None

        elif event.kind == "ACT":
            act_counter += 1
            current_act_id = f"{document_id}:act:{act_counter:04d}"
            current_act_title = event.heading

        elif event.kind == "ARTICLE":
            current_article_number = event.article_number
            if current_act_id is not None:
                current_article_id = f"{current_act_id}:article:{current_article_number}"

        elif event.kind == "DOCUMENT_SECTION":
            # Administrative divisions group acts but are not acts themselves.
            current_act_id = None
            current_act_title = None

        cursor = event.start

    if cursor < len(text):
        yield (
            current_heading,
            cursor,
            len(text),
            current_heading_type,
            current_article_number,
            current_act_id,
            current_act_title,
            current_article_id,
        )


def _split_text_with_offsets(
    text: str,
    chunk_size: int,
    overlap: int,
    prefer_boundaries: bool = True,
) -> Iterator[tuple[int, int, str]]:
    """Split text while preserving offsets and preferring natural boundaries."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Paramètres invalides : chunk_size > overlap >= 0 requis.")

    start = 0

    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        end = hard_end

        if prefer_boundaries and hard_end < len(text):
            minimum_end = start + max(1, int(chunk_size * 0.60))
            separators = ("\n\n", "\n", ". ", "; ", ": ", " ")

            for separator in separators:
                position = text.rfind(
                    separator,
                    minimum_end,
                    hard_end,
                )

                if position >= minimum_end:
                    end = position + len(separator)
                    break

        piece = text[start:end].strip()

        if piece:
            yield start, end, piece

        if end >= len(text):
            break

        next_start = max(start + 1, end - overlap)

        if prefer_boundaries:
            while next_start < end and not text[next_start].isspace():
                next_start += 1

            if next_start >= end:
                next_start = max(start + 1, end - overlap)

        start = next_start


def _fixed_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> Iterator[str]:
    """Fixed-size chunks with exact character overlap."""
    for _, _, piece in _split_text_with_offsets(
        text,
        chunk_size,
        overlap,
        prefer_boundaries=False,
    ):
        yield piece


def _semantic_chunks(
    text: str,
    chunk_size: int,
    similarity_fn: Callable[[str, str], float] | None,
    threshold: float,
) -> Iterator[str]:
    """Experimental semantic chunking using a supplied local similarity function."""
    if similarity_fn is None:
        raise ValueError("La stratégie semantic exige une similarity_fn locale.")

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    current: list[str] = []

    for sentence in sentences:
        if not sentence:
            continue

        candidate = " ".join(current + [sentence])

        if current and (
            similarity_fn(" ".join(current), sentence) < threshold or len(candidate) > chunk_size
        ):
            yield " ".join(current)
            current = [sentence]
        else:
            current.append(sentence)

    if current:
        yield " ".join(current)


def _build_chunk(
    document: LoadedDocument,
    piece: str,
    index: int,
    page_start: int | None,
    page_end: int | None,
    source_pages: list[int],
    section: str | None,
    heading_type: str | None = None,
    article_number: str | None = None,
    act_id: str | None = None,
    act_title: str | None = None,
    article_id: str | None = None,
) -> DocumentChunk:
    """Build a chunk with retrieval, legal and source-page metadata."""
    publication_date = document.metadata.publication_date

    metadata = {
        "title": document.metadata.title,
        "source_name": document.metadata.source_name,
        "source_url": document.metadata.source_url,
        "language": document.metadata.language,
        "publication_date": (publication_date.isoformat() if publication_date else None),
        "publication_year": document.metadata.publication_year,
        "document_type": document.metadata.document_type,
        "legal_reference": document.metadata.legal_reference,
        "bo_number": document.metadata.bo_number,
        "access_level": document.metadata.access_level,
        "allowed_roles": document.metadata.allowed_roles,
        "source_pages": source_pages,
        "heading_type": heading_type,
        "document_part": (
            heading_type
            if heading_type in {"FRONT_MATTER", "TABLE_OF_CONTENTS"}
            else "LEGAL_CONTENT" if heading_type is not None else None
        ),
        "article_number": article_number,
        "act_id": act_id,
        "act_title": act_title,
        "article_id": article_id,
    }

    ocr_info = document.metadata.extra.get("ocr")
    if ocr_info:
        metadata["ocr"] = ocr_info

    return DocumentChunk(
        chunk_id=stable_chunk_id(
            document.metadata.document_id,
            index,
            piece,
        ),
        document_id=document.metadata.document_id,
        text=piece,
        chunk_index=index,
        page_start=page_start,
        page_end=page_end,
        section=section,
        metadata=metadata,
    )


def chunk_document(
    document: LoadedDocument,
    strategy: str = "section",
    chunk_size: int = 1200,
    overlap: int = 180,
    semantic_similarity_threshold: float = 0.72,
    similarity_fn: Callable[[str, str], float] | None = None,
) -> list[DocumentChunk]:
    """
    Create legally structured chunks.

    Section strategy:
    - Recognizes validated article headings.
    - Rejects common financial-table false positives.
    - Separates front matter, tables of contents and legal content by page.
    - Tracks act/article identifiers without inventing fallback act IDs.
    - Ends article metadata at signatures and document headings.
    - Preserves source-page ranges.

    Fixed strategy:
    - Uses fixed character windows with exact overlap.

    Semantic strategy:
    - Uses the supplied local similarity function.
    """
    if strategy not in {"fixed", "section", "semantic"}:
        raise ValueError("strategy doit être fixed, section ou semantic.")

    chunks: list[DocumentChunk] = []
    index = 0

    if strategy == "semantic":
        for page in document.pages:
            text = _normalize_text(page.text)
            if not text:
                continue

            for piece in _semantic_chunks(
                text,
                chunk_size,
                similarity_fn,
                semantic_similarity_threshold,
            ):
                if not piece:
                    continue

                source_pages = [page.page_number] if page.page_number is not None else []

                chunks.append(
                    _build_chunk(
                        document=document,
                        piece=piece,
                        index=index,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        source_pages=source_pages,
                        section=page.section,
                    )
                )
                index += 1

        return chunks

    full_text, page_spans = _join_pages(document)
    if not full_text:
        return []

    if strategy == "fixed":
        segments = [
            (
                None,
                0,
                len(full_text),
                None,
                None,
                None,
                None,
                None,
            )
        ]
    else:
        segments = list(
            _section_ranges(
                full_text,
                document_id=document.metadata.document_id,
                page_spans=page_spans,
            )
        )

    for (
        heading,
        segment_start,
        segment_end,
        heading_type,
        article_number,
        act_id,
        act_title,
        article_id,
    ) in segments:
        section_text = full_text[segment_start:segment_end]

        for local_start, local_end, piece in _split_text_with_offsets(
            section_text,
            chunk_size,
            overlap,
            prefer_boundaries=(strategy == "section"),
        ):
            global_start = segment_start + local_start
            global_end = segment_start + local_end

            (
                page_start,
                page_end,
                source_pages,
                page_section,
            ) = _page_info(
                global_start,
                global_end,
                page_spans,
            )

            chunks.append(
                _build_chunk(
                    document=document,
                    piece=piece,
                    index=index,
                    page_start=page_start,
                    page_end=page_end,
                    source_pages=source_pages,
                    section=heading or page_section,
                    heading_type=heading_type,
                    article_number=article_number,
                    act_id=act_id,
                    act_title=act_title,
                    article_id=article_id,
                )
            )
            index += 1

    return chunks
