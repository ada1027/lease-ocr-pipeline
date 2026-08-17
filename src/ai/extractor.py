"""Lease document extractor — classifies doc type and extracts SF, tenant, suite via OpenRouter."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from openai import OpenAI

from src.utils.logging import get_logger

logger = get_logger(__name__)

MODEL = "anthropic/claude-sonnet-4-5"

SYSTEM_PROMPT = """\
You are a commercial real estate analyst extracting data from lease documents.

STEP 1 — Classify the document:
  - "executed_lease"  : a signed lease agreement for a specific tenant's space
  - "leasing_flyer"   : a marketing brochure advertising available space(s) in a center/building
  - "tenant_roster"   : a list of tenants (mall directory, rent roll, etc.)
  - "unknown"         : cannot be determined

STEP 2 — Extract fields based on doc type:

For executed_lease:
  - square_footage: the tenant's DEMISED PREMISES / leased area (not the whole building)
  - tenant_name: the tenant entity named in the lease
  - suite_number: the suite, unit, or space number
  - source_tag: "lease_ocr"

For leasing_flyer:
  - square_footage: the TOTAL CENTER / BUILDING GLA (e.g. "46,444 SF Food Lion-anchored center")
    NOT individual available suite sizes. If only suite sizes are listed and no total is given, sum
    them or use the largest stated total and note it in evidence_snippet.
  - tenant_name: the shopping center / property name (e.g. "Bainbridge Marketplace")
  - suite_number: null (flyers advertise the property, not a specific suite)
  - source_tag: "leasing_pdf"

For tenant_roster:
  - square_footage: the specific tenant's square footage if a store_id is identifiable, else the
    total GLA of the property
  - tenant_name: the anchor tenant or property name
  - suite_number: the suite associated with the primary tenant if present
  - source_tag: "mall_directory"

For unknown:
  - extract whatever square footage is most prominent
  - source_tag: "lease_ocr"

COVERAGE RULE — never return null for square_footage:
  - If no explicit SF is found, make your best estimate from context clues (room counts, parking
    ratios, comparable references) and set confidence accordingly.
  - If truly nothing can be inferred, return 0 with confidence 0.1 and explain in evidence.

Return ONLY raw JSON (no markdown, no backticks):
{
  "doc_type":    "executed_lease" | "leasing_flyer" | "tenant_roster" | "unknown",
  "square_footage": <integer — never null>,
  "unit":        "sq ft" | "sq m",
  "confidence":  <float 0.0–1.0, e.g. 0.95 = near-certain, 0.7 = probable, 0.4 = uncertain>,
  "tenant_name": <string or null>,
  "suite_number": <string or null>,
  "source_tag":  "lease_ocr" | "leasing_pdf" | "mall_directory",
  "evidence":    <exact text excerpt proving the value, max 300 chars>
}
"""


@dataclass
class ExtractionResult:
    square_footage: int           # never None — fallback to 0 with low confidence
    unit: Optional[str]
    confidence: float             # 0.0–1.0 per proposal output schema
    evidence: str                 # text excerpt or file reference proving the value
    doc_type: str = "unknown"
    tenant_name: Optional[str] = None
    suite_number: Optional[str] = None
    source_tag: str = "lease_ocr"
    raw_response: str = ""
    extracted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _strip_code_block(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def get_client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def _fallback_result(reason: str) -> ExtractionResult:
    return ExtractionResult(
        square_footage=0,
        unit=None,
        confidence=0.1,
        evidence=reason,
        doc_type="unknown",
        source_tag="lease_ocr",
    )


def extract_square_footage(candidate_text: str) -> ExtractionResult:
    """Send candidate text to the model and return a fully populated ExtractionResult."""
    if not candidate_text.strip():
        logger.warning("No candidate text — returning low-confidence fallback.")
        return _fallback_result("No extractable text found in document")

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
            square_footage=0, unit=None, confidence=0.1,
            evidence="Model returned non-JSON response",
            raw_response=raw,
        )

    sf = data.get("square_footage")
    if sf is None:
        sf = 0

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
        source_tag=data.get("source_tag", "lease_ocr"),
        raw_response=raw,
    )
