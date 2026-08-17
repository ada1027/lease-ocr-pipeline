"""Claude Vision fallback via OpenRouter — for low-confidence or fully scanned pages."""

from __future__ import annotations

import base64
import json
from typing import List

from src.ai.extractor import MODEL, ExtractionResult, get_client, _strip_code_block
from src.extraction.ocr import rasterize_page
from src.ingestion.loader import PageInfo
from src.utils.logging import get_logger

logger = get_logger(__name__)

FLOORPLAN_KEYWORDS = ["floor plan", "floor layout", "site plan", "exhibit", "schedule"]

VISION_PROMPT = """\
This is a page from a commercial real estate document — a lease, flyer, site plan, or tenant roster.
Carefully analyse all text, tables, diagrams, floor plans, and annotations visible.

Classify the document and extract square footage:
- executed_lease: return the tenant's demised premises / leased area
- leasing_flyer:  return the total center / building GLA (not individual suite sizes)
- tenant_roster:  return the GLA or the primary tenant's SF
- unknown:        return whatever SF figure is most prominent

COVERAGE RULE: never return null for square_footage. If nothing is visible, return 0.

Return ONLY raw JSON (no markdown, no backticks):
{
  "doc_type":      "executed_lease" | "leasing_flyer" | "tenant_roster" | "unknown",
  "square_footage": <integer — never null>,
  "unit":          "sq ft" | "sq m",
  "confidence":    <float 0.0–1.0>,
  "tenant_name":   <string or null>,
  "suite_number":  <string or null>,
  "evidence":      <describe exactly what you saw, max 300 chars>
}
"""


def find_floorplan_pages(pages: List[PageInfo]) -> List[int]:
    """Return 1-based page numbers that look like floorplans or diagrams."""
    candidates = [
        p.page_number for p in pages
        if any(kw in p.text.lower() for kw in FLOORPLAN_KEYWORDS)
    ]
    if not candidates:
        candidates = [p.page_number for p in pages if p.mode == "scanned"]
    logger.info("Vision candidate pages: %s", candidates)
    return candidates


def extract_via_vision(pdf_path: str, page_numbers: List[int]) -> ExtractionResult:
    """Send rasterized page images to Claude Vision via OpenRouter."""
    if not page_numbers:
        logger.warning("No pages provided for vision extraction.")
        return ExtractionResult(
            square_footage=0, unit=None, confidence=0.1,
            evidence="No pages available for vision extraction",
            source_tag="vision_estimate",
        )

    client = get_client()
    content = []

    for pn in page_numbers:
        png_bytes = rasterize_page(pdf_path, pn)
        b64 = base64.standard_b64encode(png_bytes).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
        logger.debug("Encoded page %d for vision (%d bytes).", pn, len(png_bytes))

    content.append({"type": "text", "text": VISION_PROMPT})

    logger.info("Sending %d page image(s) to Claude Vision via OpenRouter.", len(page_numbers))

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": content}],
    )

    raw = response.choices[0].message.content or ""
    logger.debug("Vision raw response: %s", raw)

    try:
        data = json.loads(_strip_code_block(raw))
    except json.JSONDecodeError:
        logger.error("Vision response was not valid JSON: %s", raw)
        return ExtractionResult(
            square_footage=0, unit=None, confidence=0.1,
            evidence="Vision model returned non-JSON response",
            source_tag="vision_estimate",
            raw_response=raw,
        )

    sf = data.get("square_footage") or 0
    raw_conf = data.get("confidence", 0.5)
    confidence = float(raw_conf) if isinstance(raw_conf, (int, float)) else 0.5

    return ExtractionResult(
        square_footage=int(sf),
        unit=data.get("unit"),
        confidence=confidence,
        evidence=data.get("evidence", data.get("evidence_snippet", "")),
        doc_type=data.get("doc_type", "unknown"),
        tenant_name=data.get("tenant_name"),
        suite_number=data.get("suite_number"),
        source_tag="vision_estimate",   # always override — vision results are always this source
        raw_response=raw,
    )
