"""Claude Vision fallback — used when text extraction confidence is low."""

from __future__ import annotations

import base64
import os
from typing import List

import anthropic

from src.ai.extractor import MODEL, ExtractionResult
from src.extraction.ocr import rasterize_page
from src.ingestion.loader import PageInfo
from src.utils.logging import get_logger

logger = get_logger(__name__)

FLOORPLAN_KEYWORDS = ["floor plan", "floor layout", "site plan", "exhibit", "schedule"]


def find_floorplan_pages(pages: List[PageInfo]) -> List[int]:
    """Return 1-based page numbers that look like floorplans."""
    candidates = []
    for page in pages:
        text_lower = page.text.lower()
        if any(kw in text_lower for kw in FLOORPLAN_KEYWORDS):
            candidates.append(page.page_number)
    logger.info("Floorplan candidate pages: %s", candidates)
    return candidates


def extract_via_vision(pdf_path: str, page_numbers: List[int]) -> ExtractionResult:
    """Send rasterized page images to Claude Vision and parse the result."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    image_blocks = []
    for pn in page_numbers:
        png_bytes = rasterize_page(pdf_path, pn)
        b64 = base64.standard_b64encode(png_bytes).decode()
        image_blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64},
            }
        )
        logger.debug("Encoded page %d for vision (%d bytes).", pn, len(png_bytes))

    image_blocks.append(
        {
            "type": "text",
            "text": (
                "These are pages from a commercial lease. "
                "Extract the primary leasable square footage from any diagrams, "
                "tables, or annotations visible. "
                "Return JSON with keys: square_footage, unit, confidence, evidence_snippet."
            ),
        }
    )

    logger.info("Sending %d page image(s) to Claude Vision.", len(page_numbers))
    message = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": image_blocks}],
    )

    raw = message.content[0].text
    import json

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Vision response non-JSON: %s", raw)
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
