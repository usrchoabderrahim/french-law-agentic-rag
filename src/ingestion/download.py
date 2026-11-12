from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests

from ingestion.schemas import SourceCandidate


def _allowed(url: str, hosts: set[str]) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and (parsed.hostname or "").lower() in hosts


def _robots_ok(url, session, user_agent, timeout, cache):
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


def download_pdf(
    candidate: SourceCandidate,
    output_dir: str | Path,
    allowed_hosts: list[str],
    user_agent: str,
    timeout: int = 30,
    delay: float = 2.0,
    max_size_mb: int = 100,
    session: requests.Session | None = None,
) -> dict:
    """Download one French SGG PDF after source, language and period checks."""
    if candidate.language != "fr" or candidate.review_reason:
        return {
            "status": "manual_review",
            "url": candidate.url,
            "reason": candidate.review_reason,
        }

    hosts = {host.lower() for host in allowed_hosts}
    if not _allowed(candidate.url, hosts):
        return {"status": "rejected_host", "url": candidate.url}

    http = session or requests.Session()
    if not _robots_ok(candidate.url, http, user_agent, timeout, {}):
        return {"status": "blocked_by_robots_or_unavailable", "url": candidate.url}

    time.sleep(max(0.0, delay))
    response = http.get(
        candidate.url,
        timeout=timeout,
        stream=True,
        allow_redirects=False,
        headers={"User-Agent": user_agent, "Accept": "application/pdf"},
    )
    if response.is_redirect:
        target = urljoin(candidate.url, response.headers.get("Location", ""))
        if not _allowed(target, hosts):
            return {
                "status": "manual_review_external_redirect",
                "url": candidate.url,
                "redirect": target,
            }
        return {"status": "manual_review_redirect", "url": candidate.url, "redirect": target}

    response.raise_for_status()
    content_length = response.headers.get("Content-Length")
    if content_length and int(content_length) > max_size_mb * 1024 * 1024:
        return {"status": "rejected_too_large", "url": candidate.url}

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    url_digest = hashlib.sha256(candidate.url.encode("utf-8")).hexdigest()[:16]
    temp_path = output / f"sgg_{candidate.detected_year or 'unknown'}_{url_digest}.part"
    final_path = temp_path.with_suffix(".pdf")

    total = 0
    signature = b""
    try:
        with temp_path.open("wb") as stream:
            for block in response.iter_content(chunk_size=128 * 1024):
                if not block:
                    continue
                if len(signature) < 8:
                    signature += block[: 8 - len(signature)]
                total += len(block)
                if total > max_size_mb * 1024 * 1024:
                    raise ValueError("Fichier supérieur à la taille maximale autorisée.")
                stream.write(block)
        if not signature.startswith(b"%PDF-"):
            raise ValueError("La signature binaire du fichier n'est pas celle d'un PDF.")
        temp_path.replace(final_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    digest = hashlib.sha256(final_path.read_bytes()).hexdigest()
    return {
        "status": "downloaded",
        "url": candidate.url,
        "local_path": str(final_path),
        "sha256": digest,
        "size_bytes": final_path.stat().st_size,
        "detected_year": candidate.detected_year,
        "language": "fr",
        "label": candidate.label,
    }


def append_jsonl(path: str | Path, row: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
