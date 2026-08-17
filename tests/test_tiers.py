"""Tests for Tier 0.5, Tier 2, and Tier 3 cascade modules."""

from __future__ import annotations

import csv
import dataclasses
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Tier 2 — Chain bounds check
# ---------------------------------------------------------------------------

@pytest.fixture
def chain_averages_csv(tmp_path):
    p = tmp_path / "chain_averages.csv"
    p.write_text("chain_name,avg_sf,min_sf,max_sf\nChipotle,2400,1800,3200\nSubway,1500,800,2200\n")
    return p


def test_tier2_loads_csv(chain_averages_csv):
    from src.tiers.tier2_bounds import load_chain_averages
    avgs = load_chain_averages(chain_averages_csv)
    assert "chipotle" in avgs
    assert avgs["chipotle"].avg_sf == 2400
    assert avgs["chipotle"].min_sf == 1800
    assert avgs["chipotle"].max_sf == 3200


def test_tier2_within_bounds(chain_averages_csv):
    from src.tiers.tier2_bounds import load_chain_averages, check_bounds
    avgs = load_chain_averages(chain_averages_csv)
    r = check_bounds("Chipotle", 2400, avgs)
    assert r.within_bounds is True
    assert r.flag == "ok"


def test_tier2_below_floor(chain_averages_csv):
    from src.tiers.tier2_bounds import load_chain_averages, check_bounds
    avgs = load_chain_averages(chain_averages_csv)
    # lower bound = 1800 * 0.30 = 540
    r = check_bounds("Chipotle", 400, avgs)
    assert r.within_bounds is False
    assert r.flag == "below_floor"


def test_tier2_above_ceiling(chain_averages_csv):
    from src.tiers.tier2_bounds import load_chain_averages, check_bounds
    avgs = load_chain_averages(chain_averages_csv)
    # upper bound = 3200 * 3.0 = 9600
    r = check_bounds("Chipotle", 20000, avgs)
    assert r.within_bounds is False
    assert r.flag == "above_ceiling"


def test_tier2_unknown_chain_does_not_block(chain_averages_csv):
    from src.tiers.tier2_bounds import load_chain_averages, check_bounds
    avgs = load_chain_averages(chain_averages_csv)
    r = check_bounds("BrandNewChain", 5000, avgs)
    # no data → don't block, just flag
    assert r.within_bounds is True
    assert r.flag == "no_chain_data"


def test_tier2_fallback_returns_average(chain_averages_csv):
    from src.tiers.tier2_bounds import load_chain_averages, fallback_from_average
    avgs = load_chain_averages(chain_averages_csv)
    assert fallback_from_average("Chipotle", avgs) == 2400
    assert fallback_from_average("Nobody", avgs) is None


def test_tier2_missing_csv_returns_empty(tmp_path):
    from src.tiers.tier2_bounds import load_chain_averages
    avgs = load_chain_averages(tmp_path / "doesnt_exist.csv")
    assert avgs == {}


def test_tier2_result_as_dict(chain_averages_csv):
    from src.tiers.tier2_bounds import load_chain_averages, check_bounds
    avgs = load_chain_averages(chain_averages_csv)
    r = check_bounds("Chipotle", 2400, avgs)
    d = r.as_dict()
    assert set(d.keys()) == {"chain_name", "avg_sf", "min_sf", "max_sf", "within_bounds", "bounds_flag"}


# ---------------------------------------------------------------------------
# Tier 3 — Overture footprint (mocked — S3 queries are slow/expensive in CI)
# ---------------------------------------------------------------------------

def _make_overture_row(building_id, area_deg2, cx, cy, xmin=None, xmax=None, ymin=None, ymax=None, height=None):
    return (building_id, area_deg2, xmin, xmax, ymin, ymax, cx, cy)


def _make_mock_duckdb(rows):
    """Return a mock duckdb module whose connect() yields rows from execute().fetchall()."""
    mock_con = MagicMock()
    mock_con.execute.return_value.fetchall.return_value = rows
    mock_duckdb = MagicMock()
    mock_duckdb.connect.return_value = mock_con
    return mock_duckdb, mock_con


def test_tier3_returns_footprint_on_match():
    from src.tiers.tier3_overture import lookup_footprint
    import sys

    mock_duckdb, _ = _make_mock_duckdb([
        ("building-abc", 1e-4, -84.39, -84.38, 33.748, 33.750, -84.388, 33.749)
    ])
    with patch.dict(sys.modules, {"duckdb": mock_duckdb}):
        result = lookup_footprint(lat=33.749, lng=-84.388, store_id="S1")

    assert result.source == "overture_footprint"
    assert result.building_id == "building-abc"
    assert result.area_sqft is not None and result.area_sqft > 0
    assert 0.0 < result.confidence <= 1.0


def test_tier3_no_match_returns_no_match():
    from src.tiers.tier3_overture import lookup_footprint
    import sys

    mock_duckdb, _ = _make_mock_duckdb([])
    with patch.dict(sys.modules, {"duckdb": mock_duckdb}):
        result = lookup_footprint(lat=33.749, lng=-84.388, store_id="S1")

    assert result.source == "no_match"
    assert result.area_sqft is None
    assert result.confidence == 0.0


def test_tier3_query_error_returns_no_match():
    from src.tiers.tier3_overture import lookup_footprint
    import sys

    mock_con = MagicMock()
    mock_con.execute.side_effect = Exception("S3 connection timeout")
    mock_duckdb = MagicMock()
    mock_duckdb.connect.return_value = mock_con

    with patch.dict(sys.modules, {"duckdb": mock_duckdb}):
        result = lookup_footprint(lat=33.749, lng=-84.388, store_id="S1")

    assert result.source == "no_match"
    assert "Query error" in result.evidence


def test_tier3_confidence_degrades_with_distance():
    from src.tiers.tier3_overture import lookup_footprint
    import sys

    mock_duckdb_close, mock_con_close = _make_mock_duckdb([
        ("b1", 1e-4, -84.39, -84.38, 33.748, 33.750, -84.38800, 33.74904)   # ~5m away
    ])
    with patch.dict(sys.modules, {"duckdb": mock_duckdb_close}):
        r_close = lookup_footprint(lat=33.749, lng=-84.388, store_id="S1")

    mock_duckdb_far, mock_con_far = _make_mock_duckdb([
        ("b2", 1e-4, -84.39, -84.38, 33.748, 33.750, -84.3884, 33.7494)     # ~40m away
    ])
    with patch.dict(sys.modules, {"duckdb": mock_duckdb_far}):
        r_far = lookup_footprint(lat=33.749, lng=-84.388, store_id="S2")

    assert r_close.confidence >= r_far.confidence


def test_tier3_returns_dataclass():
    from src.tiers.tier3_overture import lookup_footprint, FootprintResult
    import sys

    mock_duckdb, _ = _make_mock_duckdb([])
    with patch.dict(sys.modules, {"duckdb": mock_duckdb}):
        result = lookup_footprint(lat=33.749, lng=-84.388, store_id="S1")

    assert isinstance(result, FootprintResult)
    d = dataclasses.asdict(result)
    assert "store_id" in d and "area_sqft" in d and "source" in d and "confidence" in d


# ---------------------------------------------------------------------------
# Tier 0.5 — Scraper
# ---------------------------------------------------------------------------

MOCK_LEASING_PAGE = """
<html><body>
  <h1>Eastgate Shopping Center</h1>
  <a href="/docs/leasing_brochure.pdf">Download Leasing Brochure</a>
  <a href="/docs/site_plan.pdf">Site Plan</a>
  <div class="tenant-list">
    <p>Chipotle - Suite 104 - 2,350 SF</p>
    <p>Starbucks - Suite 110 - 1,800 SF</p>
  </div>
</body></html>
"""


@patch("src.tiers.tier0_5_scraper._download_pdf")
@patch("httpx.Client.get")
def test_scraper_finds_pdf_links(mock_get, mock_download, tmp_path):
    from src.tiers.tier0_5_scraper import scrape_center

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = MOCK_LEASING_PAGE
    mock_resp.content = b"<html>"
    mock_resp.headers = {"content-type": "text/html"}
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    fake_pdf = tmp_path / "leasing_brochure.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")
    mock_download.return_value = fake_pdf

    result = scrape_center("S1", "Eastgate", "https://eastgate.example.com/leasing",
                           pdf_dir=tmp_path)
    assert len(result.pdfs_found) == 2
    assert any("leasing_brochure" in p for p in result.pdfs_found)


@patch("src.tiers.tier0_5_scraper._download_pdf")
@patch("httpx.Client.get")
def test_scraper_extracts_sf_from_directory(mock_get, mock_download, tmp_path):
    from src.tiers.tier0_5_scraper import scrape_center

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = MOCK_LEASING_PAGE
    mock_resp.content = b"<html>"
    mock_resp.headers = {"content-type": "text/html"}
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp
    mock_download.return_value = None

    result = scrape_center("S1", "Eastgate", "https://eastgate.example.com/leasing",
                           chain_name="Chipotle", pdf_dir=tmp_path)
    assert result.sf_from_directory == 2350
    assert result.tenant_name == "Chipotle"


@patch("httpx.Client.get")
def test_scraper_handles_http_error(mock_get, tmp_path):
    import httpx
    from src.tiers.tier0_5_scraper import scrape_center

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not Found", request=MagicMock(), response=mock_resp
    )
    mock_get.return_value = mock_resp

    result = scrape_center("S1", "Bad Center", "https://bad.example.com",
                           pdf_dir=tmp_path)
    assert result.status == "error"
    assert "404" in result.error


@patch("httpx.Client.get")
def test_scraper_detects_direct_pdf_url(mock_get, tmp_path):
    from src.tiers.tier0_5_scraper import scrape_center

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"%PDF-1.4 fake pdf content"
    mock_resp.headers = {"content-type": "application/pdf"}
    mock_resp.text = ""
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    result = scrape_center("S1", "Direct PDF", "https://example.com/flyer.pdf",
                           pdf_dir=tmp_path)
    # Should download the PDF directly, not try to parse HTML
    assert result.pdfs_found == ["https://example.com/flyer.pdf"]


def test_scraper_missing_centers_csv_raises(tmp_path):
    from src.tiers.tier0_5_scraper import run_batch
    with pytest.raises(FileNotFoundError):
        run_batch(centers_path=tmp_path / "missing.csv")


# ---------------------------------------------------------------------------
# Live Tier 3 test — skipped unless explicitly enabled
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("RUN_LIVE_TIER3"),
    reason="Skipped by default — set RUN_LIVE_TIER3=1 to run (costs S3 data transfer)"
)
def test_tier3_live_overture_query():
    from src.tiers.tier3_overture import lookup_footprint
    # Known building in downtown Atlanta
    result = lookup_footprint(lat=33.7490, lng=-84.3880, store_id="LIVE-TEST")
    assert result.source in ("overture_footprint", "no_match")
    if result.source == "overture_footprint":
        assert result.area_sqft and result.area_sqft > 0
