"""Text extraction for searchable pages — filters for GLA / sq ft / demised premises."""

from __future__ import annotations

import re
from typing import List

from src.ingestion.loader import PageInfo
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Patterns that indicate a square-footage clause worth sending to the AI step.
_SIGNAL_PATTERNS = [
    re.compile(r"\bGLA\b"),
    re.compile(r"\bsq\.?\s*ft\.?\b", re.IGNORECASE),
    re.compile(r"\bsquare\s+feet\b", re.IGNORECASE),
    re.compile(r"\bsquare\s+foot\b", re.IGNORECASE),
    re.compile(r"\bdemised\s+premises\b", re.IGNORECASE),
    re.compile(r"\bleasable\s+area\b", re.IGNORECASE),
    re.compile(r"\brentable\s+area\b", re.IGNORECASE),
    re.compile(r"\bfor\s+lease\b", re.IGNORECASE),
    re.compile(r"\bSF\b"),                          # bare "SF" column / label
    re.compile(r"\b\d[\d,]+\s*SF\b"),              # e.g. "1,678 SF"
]


def extract_candidate_text(pages: List[PageInfo]) -> str:
    """Concatenate text from searchable pages that contain GLA/sqft signals."""
    candidates: list[str] = []
    for page in pages:
        if page.mode != "searchable":
            continue
        if any(pat.search(page.text) for pat in _SIGNAL_PATTERNS):
            logger.debug("Page %d matched signal pattern — included.", page.page_number)
            candidates.append(f"--- Page {page.page_number} ---\n{page.text.strip()}")
        else:
            logger.debug("Page %d: no signals found — skipped.", page.page_number)

    logger.info(
        "Candidate pages from searchable extraction: %d / %d",
        len(candidates),
        sum(1 for p in pages if p.mode == "searchable"),
    )
    return "\n\n".join(candidates)
