from __future__ import annotations

import re
import time
from datetime import UTC, date, datetime
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from ingestion.schemas import SourceCandidate

YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
SGG_TABLE_PATH = "/DesktopModules/MVC/TableListBO/BO/AjaxMethod"


def clean_url(url: str) -> str:
    return urldefrag(url.strip())[0]


def is_sgg_host_allowed(url: str, allowed_hosts: list[str]) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and (parsed.hostname or "").lower() in {
        host.lower() for host in allowed_hosts
    }


def _year_from_text(value: str) -> int | None:
    matches = YEAR_RE.findall(value or "")
    return int(matches[-1]) if matches else None


def _robots_allowed(url, session, user_agent, timeout, cache) -> bool:
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin not in cache:
        try:
            response = session.get(
                urljoin(origin, "/robots.txt"),
                timeout=timeout,
                headers={"User-Agent": user_agent},
            )
            parser = RobotFileParser()
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
            elif response.status_code in (404, 410):
                parser.parse(["User-agent: *", "Allow: /"])
            else:
                return False
            cache[origin] = parser
        except requests.RequestException:
            return False
    return cache[origin].can_fetch(user_agent, url)


def _safe_get(
    url, allowed_hosts, session, user_agent, timeout, delay, robots_cache, *, ajax_headers=None
):
    headers = {"User-Agent": user_agent, "Accept-Language": "fr-FR,fr;q=0.9"}
    headers.update(ajax_headers or {})
    current = url
    for _ in range(5):
        if not is_sgg_host_allowed(current, list(allowed_hosts)):
            raise ValueError(f"URL ou hôte non autorisé : {current}")
        if not _robots_allowed(current, session, user_agent, timeout, robots_cache):
            raise PermissionError(f"robots.txt interdit ou non vérifiable : {current}")
        time.sleep(max(0.0, delay))
        response = session.get(
            current,
            timeout=timeout,
            allow_redirects=False,
            headers=headers,
        )
        if response.is_redirect:
            if ajax_headers:
                raise RuntimeError("Le service du catalogue SGG a renvoyé une redirection.")
            location = response.headers.get("Location")
            if not location:
                raise RuntimeError(f"Redirection sans Location : {current}")
            current = urljoin(current, location)
            continue
        response.raise_for_status()
        return response
    raise RuntimeError("Trop de redirections successives.")


def _french_table_headers(soup: BeautifulSoup) -> dict[str, str] | None:
    for anchor in soup.select('a[href^="#"]'):
        label = " ".join(anchor.stripped_strings).casefold()
        if label not in ("édition de traduction officielle", "edition de traduction officielle"):
            continue
        tab = soup.find(id=anchor["href"][1:])
        if tab is None:
            continue
        for script in tab.find_all("script"):
            source = script.get_text()
            if SGG_TABLE_PATH.lstrip("/") not in source:
                continue
            module_id = re.search(r"\bModuleId\s*=\s*(\d+)\s*;", source)
            tab_id = re.search(r"\bTabId\s*=\s*(\d+)\s*;", source)
            token_input = soup.find("input", attrs={"name": "__RequestVerificationToken"})
            token = token_input.get("value") if token_input else None
            if not (module_id and tab_id and token):
                raise ValueError("Paramètres du tableau français manquants dans le catalogue SGG.")
            return {
                "ModuleId": module_id[1],
                "TabId": tab_id[1],
                "RequestVerificationToken": token,
                "X-Requested-With": "XMLHttpRequest",
                "Accept": "application/json",
            }
    return None


def _bulletin_year(bulletin_date: str, pdf_url: str) -> int | None:
    # ASP.NET MVC can serialize dates as milliseconds since the Unix epoch.
    timestamp = re.fullmatch(r"/Date\((-?\d+)(?:[+-]\d{4})?\)/", bulletin_date)
    if timestamp:
        try:
            return datetime.fromtimestamp(int(timestamp[1]) / 1000, tz=UTC).year
        except (ValueError, OverflowError, OSError):
            pass
    year = _year_from_text(bulletin_date) if timestamp is None else None
    if year is not None:
        return year
    path_year = re.search(r"/(19\d{2}|20\d{2})/", urlparse(pdf_url).path)
    return int(path_year[1]) if path_year else None


def _review_reason(language: str | None, year: int | None, cutoff_year: int) -> str | None:
    if language is None:
        return "Langue française non identifiable dans le chemin ou le libellé."
    if year is None:
        return "Année non identifiable ; vérifier la date sur le document."
    if year < cutoff_year:
        return f"Hors fenêtre récente (année détectée : {year})."
    return None


def discover_sgg_pdf_links(
    catalogue_url: str,
    allowed_hosts: list[str],
    recent_years: int = 10,
    user_agent: str = "MA-Regulatory-Assistant-Research/0.1",
    timeout: int = 30,
    delay: float = 2.0,
    session: requests.Session | None = None,
) -> tuple[str, list[SourceCandidate], str]:
    """Discover PDFs from the French edition table and static catalogue links."""
    http = session or requests.Session()
    robots_cache = {}
    response = _safe_get(
        catalogue_url, set(allowed_hosts), http, user_agent, timeout, delay, robots_cache
    )
    content_type = response.headers.get("Content-Type", "").lower()
    if "html" not in content_type and "text/" not in content_type:
        raise ValueError(f"La page catalogue n'est pas du HTML : {content_type}")

    soup = BeautifulSoup(response.text, "html.parser")
    candidates: dict[str, SourceCandidate] = {}
    cutoff_year = date.today().year - recent_years

    for anchor in soup.select("a[href]"):
        href = (anchor.get("href") or "").strip()
        if not href or href.lower().startswith(("javascript:", "mailto:", "tel:")):
            continue
        absolute = clean_url(urljoin(response.url, href))
        if not is_sgg_host_allowed(absolute, allowed_hosts):
            continue

        label = " ".join(anchor.get_text(" ", strip=True).split())
        path = urlparse(absolute).path.lower()
        if not path.endswith(".pdf"):
            continue

        french_marker = "/bo/fr/" in path or "/bo/bo_fr/" in path
        french_label = any(
            marker in label.lower()
            for marker in (
                "édition de traduction officielle",
                "edition de traduction officielle",
                "français",
                "française",
            )
        )
        language = "fr" if french_marker or french_label else None
        year = _year_from_text(f"{label} {absolute}")

        candidates[absolute] = SourceCandidate(
            url=absolute,
            label=label,
            source_page=response.url,
            detected_year=year,
            language=language,
            review_reason=_review_reason(language, year, cutoff_year),
        )

    headers = _french_table_headers(soup)
    if headers is not None:
        table = _safe_get(
            urljoin(response.url, SGG_TABLE_PATH),
            set(allowed_hosts),
            http,
            user_agent,
            timeout,
            delay,
            robots_cache,
            ajax_headers={**headers, "Referer": response.url},
        )
        rows = table.json()
        if not isinstance(rows, list):
            raise ValueError("Format inattendu pour le tableau français du catalogue SGG.")
        for row in rows:
            href = row.get("BoUrl") if isinstance(row, dict) else None
            if not isinstance(href, str) or not href.strip():
                continue
            absolute = clean_url(urljoin(response.url, href.strip()))
            if not is_sgg_host_allowed(absolute, allowed_hosts):
                continue
            if not urlparse(absolute).path.lower().endswith(".pdf"):
                continue
            year = _bulletin_year(str(row.get("BoDate") or ""), absolute)
            label = (
                f"Bulletin officiel n° {row['BoNum']}" if row.get("BoNum") else "Bulletin officiel"
            )
            candidates[absolute] = SourceCandidate(
                url=absolute,
                label=label,
                source_page=response.url,
                detected_year=year,
                language="fr",
                review_reason=_review_reason("fr", year, cutoff_year),
            )

    return response.url, list(candidates.values()), response.text
