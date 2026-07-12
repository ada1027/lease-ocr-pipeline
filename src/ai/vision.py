"""Claude Vision fallback via OpenRouter — for low-confidence or fully scanned pages."""

from __future__ import annotations

import base64
import json
import os
from typing import List

from openai import OpenAI

from src.ai.extractor import MODEL, ExtractionResult
from src.extraction.ocr import rasterize_page
from src.ingestion.loader import PageInfo
from src.utils.logging import get_logger

logger = get_logger(__name__)

FLOORPLAN_KEYWORDS = ["floor plan", "floor layout", "site plan", "exhibit", "schedule"]


def find_floorplan_pages(pages: List[PageInfo]) -> List[int]:
    """Return 1-based page numbers that look like floorplans or diagrams."""
    candidates = []
    for page in pages:
        if any(kw in page.text.lower() for kw in FLOORPLAN_KEYWORDS):
            candidates.append(page.page_number)
    # If no keyword matches, fall back to all scanned pages
    if not candidates:
        candidates = [p.page_number for p in pages if p.mode == "scanned"]
    logger.info("Vision candidate pages: %s", candidates)
    return candidates


def extract_via_vision(pdf_path: str, page_numbers: List[int]) -> ExtractionResult:
    """Send rasterized page images to Claude Vision via OpenRouter."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    content = []
    for pn in page_numbers:
        png_bytes = rasterize_page(pdf_path, pn)
        b64 = base64.standard_b64encode(png_bytes).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
        logger.debug("Encoded page %d for vision (%d bytes).", pn, len(png_bytes))

    content.append({
        "type": "text",
        "text": (
            "These are pages from a commercial lease or property flyer. "
            "Extract the primary leasable square footage from any text, "
            "tables, diagrams, or annotations visible. "
            "Return ONLY valid JSON with keys: "
            "square_footage (integer or null), unit (\"sq ft\" or null), "
            "confidence (\"high\"/\"medium\"/\"low\"), "
            "evidence_snippet (what you saw, 200 chars max)."
        ),
    })

    logger.info("Sending %d page image(s) to Claude Vision via OpenRouter.", len(page_numbers))

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": content}],
    )

    raw = response.choices[0].message.content
    logger.debug("Vision raw response: %s", raw)

    try:
        data = json.loads(raw)
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
