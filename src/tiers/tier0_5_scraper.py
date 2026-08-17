"""Tier 0.5 — Browser-harness leasing document scraper.

Finds leasing PDFs and tenant directory pages from public mall/operator websites.
Boss confirmed we have a list of shopping centers to work from.

Sources:
  - Mall operator portals (Simon, Brookfield, URW, Macerich, Tanger, etc.)
  - Property-specific leasing pages with downloadable PDFs
  - Tenant directory pages that list SF per unit

Input:  data/shopping_centers.csv  (columns: store_id, center_name, url, operator)
Output: Downloaded PDFs → data/scraped_pdfs/<store_id>/
        Enrichment rows written to data/output/enrichment_table.csv with
        source_tag "leasing_pdf" or "mall_directory"

The scraper is intentionally simple: fetch the page, find PDF links and tenant
tables, download, then hand off to the existing Tier 1 pipeline.
No headless browser needed for most operator sites — plain HTTP works.
If a site needs JS rendering, set USE_PLAYWRIGHT=1 in .env (requires
`pip install playwright && playwright install chromium`).
"""

from __future__ import annotations

import csv
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CENTERS_PATH = Path("data/shopping_centers.csv")
DEFAULT_PDF_DIR = Path("data/scraped_pdfs")
REQUEST_DELAY_S = 1.5   # be polite — don't hammer servers
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


@dataclass
class ScrapeResult:
    store_id: str
    center_name: str
    url: str
    pdfs_found: list[str] = field(default_factory=list)
    pdfs_downloaded: list[Path] = field(default_factory=list)
    sf_from_directory: Optional[int] = None
    tenant_name: Optional[str] = None
    status: str = "ok"   # "ok" | "error" | "no_content"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Operator-specific URL patterns
# (add new operators to this dict — key is a substring of the domain)
# ---------------------------------------------------------------------------

OPERATOR_PATTERNS: dict[str, dict] = {
    "simon.com": {
        "tenant_table_selector": "table.tenant-list, div.tenants",
        "pdf_link_pattern": r"\.pdf",
    },
    "brookfieldproperties.com": {
        "tenant_table_selector": "div.store-list",
        "pdf_link_pattern": r"\.pdf",
    },
    "macerich.com": {
        "tenant_table_selector": "div.store-directory",
        "pdf_link_pattern": r"leasing.*\.pdf|brochure.*\.pdf",
    },
    "tangeroutlet.com": {
        "tenant_table_selector": "div.store-directory-list",
        "pdf_link_pattern": r"\.pdf",
    },
    # Default — works for most property-specific leasing pages
    "_default": {
        "tenant_table_selector": "table, div[class*='tenant'], div[class*='store']",
        "pdf_link_pattern": r"leasing|brochure|flyer|available|spaces|\.pdf",
    },
}


def _get_operator_pattern(url: str) -> dict:
    domain = urlparse(url).netloc.lower()
    for key, pattern in OPERATOR_PATTERNS.items():
        if key != "_default" and key in domain:
            return pattern
    return OPERATOR_PATTERNS["_default"]


def _find_pdf_links(soup: BeautifulSoup, base_url: str, pattern: str) -> list[str]:
    """Return absolute URLs for all PDF links matching pattern."""
    pdf_re = re.compile(pattern, re.IGNORECASE)
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        if pdf_re.search(href) or href.lower().endswith(".pdf"):
            links.append(urljoin(base_url, href))
    return list(dict.fromkeys(links))   # deduplicate, preserve order


def _extract_sf_from_table(soup: BeautifulSoup, selector: str, chain_name: str) -> Optional[int]:
    """Try to find an SF number in a tenant directory table for a specific chain."""
    chain_lower = chain_name.lower()
    sf_re = re.compile(r"([\d,]+)\s*(?:sf|sq\.?\s*ft|square\s*feet)", re.IGNORECASE)

    for el in soup.select(selector):
        text = el.get_text(" ", strip=True)
        if chain_lower in text.lower():
            match = sf_re.search(text)
            if match:
                try:
                    return int(match.group(1).replace(",", ""))
                except ValueError:
                    pass
    return None


def _download_pdf(url: str, dest_dir: Path, client: httpx.Client) -> Optional[Path]:
    """Download a PDF to dest_dir. Returns local path or None on failure."""
    filename = Path(urlparse(url).path).name or "leasing.pdf"
    dest = dest_dir / filename
    if dest.exists():
        logger.info("PDF already downloaded: %s", dest)
        return dest
    try:
        resp = client.get(url, follow_redirects=True, timeout=30)
        if resp.status_code == 200 and b"%PDF" in resp.content[:10]:
            dest.write_bytes(resp.content)
            logger.info("Downloaded %s → %s (%d bytes)", url, dest, len(resp.content))
            return dest
        else:
            logger.warning("Non-PDF or bad status at %s: %d", url, resp.status_code)
    except Exception as e:
        logger.warning("Failed to download %s: %s", url, e)
    return None


def scrape_center(
    store_id: str,
    center_name: str,
    url: str,
    chain_name: str = "",
    pdf_dir: Path = DEFAULT_PDF_DIR,
    client: Optional[httpx.Client] = None,
) -> ScrapeResult:
    """Scrape one shopping center URL for leasing PDFs and tenant SF."""
    result = ScrapeResult(store_id=store_id, center_name=center_name, url=url)
    dest_dir = pdf_dir / store_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    own_client = client is None
    if own_client:
        client = httpx.Client(headers=HEADERS, follow_redirects=True, timeout=20)

    try:
        logger.info("Scraping store=%s url=%s", store_id, url)
        resp = client.get(url)
        resp.raise_for_status()

        # If the URL itself is a PDF, download it directly — no need to parse HTML
        content_type = resp.headers.get("content-type", "")
        if "pdf" in content_type or url.lower().endswith(".pdf") or resp.content[:4] == b"%PDF":
            path = _download_pdf(url, dest_dir, client)
            if path:
                result.pdfs_found = [url]
                result.pdfs_downloaded = [path]
            else:
                result.status = "no_content"
            return result

        soup = BeautifulSoup(resp.text, "html.parser")
        pattern = _get_operator_pattern(url)

        # 1. Find and download PDFs
        pdf_urls = _find_pdf_links(soup, url, pattern["pdf_link_pattern"])
        result.pdfs_found = pdf_urls
        logger.info("Found %d PDF links at %s", len(pdf_urls), url)

        for pdf_url in pdf_urls[:5]:    # cap at 5 per center
            time.sleep(REQUEST_DELAY_S)
            path = _download_pdf(pdf_url, dest_dir, client)
            if path:
                result.pdfs_downloaded.append(path)

        # 2. Try to extract SF directly from a tenant table
        if chain_name:
            result.sf_from_directory = _extract_sf_from_table(
                soup, pattern["tenant_table_selector"], chain_name
            )
            if result.sf_from_directory:
                result.tenant_name = chain_name
                logger.info("Found SF=%d for '%s' in tenant directory", result.sf_from_directory, chain_name)

        if not result.pdfs_downloaded and result.sf_from_directory is None:
            result.status = "no_content"

    except httpx.HTTPStatusError as e:
        result.status = "error"
        result.error = f"HTTP {e.response.status_code}"
        logger.warning("HTTP error scraping %s: %s", url, result.error)
    except Exception as e:
        result.status = "error"
        result.error = str(e)
        logger.error("Error scraping %s: %s", url, e)
    finally:
        if own_client:
            client.close()

    return result


def run_batch(
    centers_path: Path = DEFAULT_CENTERS_PATH,
    pdf_dir: Path = DEFAULT_PDF_DIR,
    chain_filter: Optional[str] = None,
) -> list[ScrapeResult]:
    """Scrape every center in shopping_centers.csv.

    CSV columns: store_id, center_name, url, operator, chain_name (optional)
    Drop your list at data/shopping_centers.csv to use this.
    """
    if not centers_path.exists():
        raise FileNotFoundError(
            f"Shopping center list not found at {centers_path}.\n"
            "Create a CSV with columns: store_id, center_name, url, operator"
        )

    results = []
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=20) as client:
        with centers_path.open() as fh:
            for row in csv.DictReader(fh):
                if chain_filter and chain_filter.lower() not in row.get("chain_name", "").lower():
                    continue
                time.sleep(REQUEST_DELAY_S)
                result = scrape_center(
                    store_id=row["store_id"],
                    center_name=row["center_name"],
                    url=row["url"],
                    chain_name=row.get("chain_name", ""),
                    pdf_dir=pdf_dir,
                    client=client,
                )
                results.append(result)
                logger.info(
                    "Scraped %s: %d PDFs downloaded, sf_from_dir=%s, status=%s",
                    row["store_id"], len(result.pdfs_downloaded),
                    result.sf_from_directory, result.status,
                )

    ok      = sum(1 for r in results if r.status == "ok")
    no_cont = sum(1 for r in results if r.status == "no_content")
    errors  = sum(1 for r in results if r.status == "error")
    pdfs    = sum(len(r.pdfs_downloaded) for r in results)

    logger.info(
        "Scrape complete: %d centers | %d ok | %d no_content | %d errors | %d PDFs downloaded",
        len(results), ok, no_cont, errors, pdfs,
    )
    return results
