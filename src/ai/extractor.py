"""Claude-based square footage extractor — text path."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import anthropic

from src.utils.logging import get_logger

logger = get_logger(__name__)

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """\
You are a commercial real estate lease analyst. Extract the primary leasable square footage \
from the lease text provided. Prioritize clauses from the "Demised Premises" or "Premises" \
section over exhibits, riders, or schedules.

Return ONLY valid JSON with these keys:
  square_footage  – integer or null
  unit            – "sq ft" | "sq m" | null
  confidence      – "high" | "medium" | "low"
  evidence_snippet – the verbatim sentence(s) you relied on (≤200 chars)
"""


@dataclass
class ExtractionResult:
    square_footage: Optional[int]
    unit: Optional[str]
    confidence: str
    evidence_snippet: str
    raw_response: str = ""


def extract_square_footage(candidate_text: str) -> ExtractionResult:
    """Send candidate text to Claude and parse the structured response."""
    if not candidate_text.strip():
        logger.warning("No candidate text to send to Claude.")
        return ExtractionResult(
            square_footage=None,
            unit=None,
            confidence="low",
            evidence_snippet="",
        )

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    logger.info("Sending %d chars to Claude (%s).", len(candidate_text), MODEL)
    message = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": candidate_text}],
    )

    raw = message.content[0].text
    logger.debug("Claude raw response: %s", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Claude returned non-JSON: %s", raw)
        return ExtractionResult(
            square_footage=None,
            unit=None,
            confidence="low",
            evidence_snippet="",
            raw_response=raw,
        )

    return ExtractionResult(
        square_footage=data.get("square_footage"),
        unit=data.get("unit"),
        confidence=data.get("confidence", "low"),
        evidence_snippet=data.get("evidence_snippet", ""),
        raw_response=raw,
    )
