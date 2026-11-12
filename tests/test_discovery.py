import json
from datetime import UTC, date, datetime
from unittest.mock import Mock

import pytest
from requests import Response, Session

from ingestion.discovery import discover_sgg_pdf_links, is_sgg_host_allowed

CATALOGUE_URL = "https://www.sgg.gov.ma/Bulletinofficiel.aspx"
TABLE_URL = "https://www.sgg.gov.ma/DesktopModules/MVC/TableListBO/BO/AjaxMethod"
CATALOGUE_HTML = """
<a href="#international">Édition des conventions internationales</a>
<div id="international"><script>
    var ModuleId = 3076;
    var TabId = 797;
    var url = "DesktopModules/MVC/TableListBO/BO/AjaxMethod";
</script></div>
<a href="#french">Édition de traduction officielle</a>
<div id="french"><script>
    var ModuleId = 2873;
    var TabId = 775;
    var url = "DesktopModules/MVC/TableListBO/BO/AjaxMethod";
</script></div>
<input name="__RequestVerificationToken" type="hidden" value="test-token">
"""


def _response(url, body, content_type="text/html; charset=utf-8"):
    response = Response()
    response.status_code = 200
    response.url = url
    response.headers["Content-Type"] = content_type
    response.encoding = "utf-8"
    response._content = body.encode("utf-8")
    return response


def test_sgg_host_allowlist():
    hosts = ["www.sgg.gov.ma", "sgg.gov.ma"]
    assert is_sgg_host_allowed(CATALOGUE_URL, hosts)
    assert not is_sgg_host_allowed("https://example.com/file.pdf", hosts)
    assert not is_sgg_host_allowed("http://www.sgg.gov.ma/file.pdf", hosts)


def test_sgg_discovers_only_public_french_table_and_keeps_static_links():
    current_year = date.today().year
    milliseconds = int(datetime(current_year, 5, 1, tzinfo=UTC).timestamp() * 1000)
    recent_pdf = "/BO/FR/2873/BO_7440_fr.pdf"
    html = (
        CATALOGUE_HTML
        + f"""
        <a href="{recent_pdf}">Télécharger</a>
        <a href="/BO/bo_fr/1956/bo_1957_fr.pdf">Archive 1956</a>
    """
    )
    rows = [
        {"BoNum": "7440", "BoDate": f"/Date({milliseconds})/", "BoUrl": recent_pdf},
        {"BoNum": "2240", "BoDate": "1955-09-30", "BoUrl": "/BO/bo_fr/1955/bo_2240_fr.pdf"},
        {"BoNum": "2000", "BoUrl": "/BO/FR/2873/BO_2000_fr.pdf"},
        {"BoUrl": "https://example.com/elsewhere.pdf"},
        {"BoUrl": "http://www.sgg.gov.ma/insecure.pdf"},
        {"BoUrl": "/not-a-pdf"},
    ]
    session = Mock(spec=Session)
    session.get.side_effect = [
        _response("https://www.sgg.gov.ma/robots.txt", "User-agent: *\nAllow: /"),
        _response(CATALOGUE_URL, html),
        _response(TABLE_URL, json.dumps(rows), "application/json"),
    ]

    final_url, candidates, snapshot = discover_sgg_pdf_links(
        CATALOGUE_URL, ["www.sgg.gov.ma"], delay=0, session=session
    )

    assert final_url == CATALOGUE_URL
    assert snapshot == html
    assert [candidate.detected_year for candidate in candidates] == [current_year, 1956, 1955, None]
    assert all(candidate.language == "fr" for candidate in candidates)
    assert candidates[0].label == "Bulletin officiel n° 7440"
    assert candidates[0].review_reason is None
    assert candidates[2].review_reason.startswith("Hors fenêtre récente")
    assert candidates[3].review_reason.startswith("Année non identifiable")
    assert session.get.call_count == 3
    args, kwargs = session.get.call_args
    assert args == (TABLE_URL,)
    assert kwargs["headers"]["ModuleId"] == "2873"
    assert kwargs["headers"]["TabId"] == "775"
    assert kwargs["headers"]["RequestVerificationToken"] == "test-token"


def test_sgg_does_not_forward_table_headers_to_a_redirect():
    redirect = _response(TABLE_URL, "")
    redirect.status_code = 302
    redirect.headers["Location"] = "https://sgg.gov.ma/redirect"
    session = Mock(spec=Session)
    session.get.side_effect = [
        _response("https://www.sgg.gov.ma/robots.txt", "User-agent: *\nAllow: /"),
        _response(CATALOGUE_URL, CATALOGUE_HTML),
        redirect,
    ]

    with pytest.raises(RuntimeError, match="redirection"):
        discover_sgg_pdf_links(
            CATALOGUE_URL, ["www.sgg.gov.ma", "sgg.gov.ma"], delay=0, session=session
        )

    assert session.get.call_count == 3
