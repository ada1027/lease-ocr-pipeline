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
You are a commercial real estate analyst. Extract the primary square footage from the document.

Rules:
- Accept any square footage figure: total center size, GLA, demised premises, suite size, or building size
- If multiple sizes appear, pick the largest or most prominent one
- Marketing flyers and tenant rosters are valid sources — use the total center/building size if present
- "221,239 square foot" or "1,200 SF" are both valid

Return ONLY raw JSON — no markdown, no code blocks, no backticks:
  square_footage   – integer or null
  unit             – "sq ft" | "sq m" | null
  confidence       – "high" | "medium" | "low"
  evidence_snippet – the exact text or description of what you found (200 chars max)
"""


@dataclass
class ExtractionResult:
    square_footage: Optional[int]
    unit: Optional[str]
    confidence: str
    evidence_snippet: str
    raw_response: str = ""


def _strip_code_block(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers Claude sometimes adds."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]   # remove first line (```json)
        text = text.rsplit("```", 1)[0]  # remove trailing ```
    return text.strip()


def get_client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def extract_square_footage(candidate_text: str) -> ExtractionResult:
    """Send candidate text to Claude via OpenRouter and return structured result."""
    if not candidate_text.strip():
        logger.warning("No candidate text to send — skipping AI extraction.")
        return ExtractionResult(
            square_footage=None, unit=None, confidence="low", evidence_snippet=""
        )

    client = get_client()
    logger.info("Sending %d chars to %s via OpenRouter.", len(candidate_text), MODEL)

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": candidate_text},
        ],
    )

    raw = response.choices[0].message.content or ""
    logger.debug("Raw response: %s", raw)

    try:
        data = json.loads(_strip_code_block(raw))
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
