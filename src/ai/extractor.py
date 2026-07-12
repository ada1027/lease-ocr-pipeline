"""Square footage extractor — sends candidate text to Claude via OpenRouter."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from src.utils.logging import get_logger

logger = get_logger(__name__)

MODEL = "anthropic/claude-sonnet-4-5"

SYSTEM_PROMPT = """\
You are a commercial real estate lease analyst. Extract the primary leasable square footage \
from the lease text provided. Prioritize clauses from the "Demised Premises" or "Premises" \
section over exhibits, riders, or schedules.

Return ONLY valid JSON with these keys:
  square_footage   – integer or null
  unit             – "sq ft" | "sq m" | null
  confidence       – "high" | "medium" | "low"
  evidence_snippet – the verbatim sentence(s) you relied on (200 chars max)
"""


@dataclass
class ExtractionResult:
    square_footage: Optional[int]
    unit: Optional[str]
    confidence: str
    evidence_snippet: str
    raw_response: str = ""


def extract_square_footage(candidate_text: str) -> ExtractionResult:
    """Send candidate text to Claude via OpenRouter and return structured result."""
    if not candidate_text.strip():
        logger.warning("No candidate text to send — skipping AI extraction.")
        return ExtractionResult(
            square_footage=None, unit=None, confidence="low", evidence_snippet=""
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    logger.info("Sending %d chars to %s via OpenRouter.", len(candidate_text), MODEL)

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": candidate_text},
        ],
    )

    raw = response.choices[0].message.content
    logger.debug("Raw response: %s", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Response was not valid JSON: %s", raw)
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
