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

Extract ANY square footage figure you can find. In order of preference:
1. Total center or building size (e.g. "221,239 sq ft shopping center")
2. The largest single space size visible
3. Any individual suite or unit size

Return ONLY raw JSON — no markdown, no code blocks, no backticks:
  square_footage   – integer or null
  unit             – "sq ft" | "sq m" | null
  confidence       – "high" | "medium" | "low"
  evidence_snippet – describe exactly what you saw (200 chars max)
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
            square_footage=None, unit=None, confidence="low", evidence_snippet=""
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
            square_footage=None, unit=None, confidence="low",
            evidence_snippet="", raw_response=raw,
        )

    return ExtractionResult(
        square_footage=data.get("square_footage"),
        unit=data.get("unit"),
        confidence=data.get("confidence", "low"),
        evidence_snippet=data.get("evidence_snippet", ""),
        raw_response=raw,
    )
