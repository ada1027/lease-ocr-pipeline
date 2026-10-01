"""
Vision benchmark — single-pass OCR+extraction without Tesseract.

Tests per-unit extraction (tenant_name, square_footage, unit_number, …) across
six models and four metrics:
  1. JSON validity rate
  2. Cross-model hallucination / within-model duplication rate
  3. Per-unit numeric accuracy vs ground truth (±10%)
  4. Cost per document

All models route through OpenRouter (OPENROUTER_API_KEY). No Anthropic SDK needed.

Models:
  claude-haiku-4-5      via OpenRouter   (rasterized pages)
  claude-sonnet-5       via OpenRouter   (rasterized pages)
  gemini-2.5-pro        via OpenRouter   (native PDF as base64 application/pdf)
  gemini-2.5-flash-lite via OpenRouter   (native PDF as base64 application/pdf)
  qwen2.5-vl-72b        via OpenRouter   (rasterized pages)
  qwen3-vl-32b          via OpenRouter   (rasterized pages)

Usage:
    python benchmark/vision_run.py
    python benchmark/vision_run.py --models haiku sonnet gemini-pro
    python benchmark/vision_run.py --pdf path/to/file.pdf
    python benchmark/vision_run.py --max-pages 20
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai.extractor import _strip_code_block
from src.extraction.ocr import rasterize_page

# ---------------------------------------------------------------------------
# Test set — same 15 PDFs as the original benchmark + 2 explicit mall directories
# ---------------------------------------------------------------------------
BASE = Path("data/SiteFiles_20260625_part01/SiteFiles")

TEST_PDFS = [
    # Single-tenant leases (small SF, ground truth known)
    str(BASE / "1150/20240912103837517060.pdf"),     # 3pp, GT=1200
    str(BASE / "10439/20240912132106205145.pdf"),     # 4pp, GT=1400
    str(BASE / "10713/20240912132524295488.pdf"),     # 2pp, GT=1400
    str(BASE / "1001/20240912104716746251.pdf"),      # 9pp, GT=unknown
    str(BASE / "10103/20240918142905086890.pdf"),     # 4pp, GT=97535
    str(BASE / "10195/20240918134820671739.pdf"),     # 4pp, GT=67019
    str(BASE / "10977/20240912133001408947.pdf"),     # 7pp, GT=46000
    str(BASE / "10800/20240912132708411685.pdf"),     # 4pp, GT=unknown
    # Multi-tenant / mall directories — required for hallucination metric
    "data/village_of_blaine.pdf",                    # GT=221239
    "data/Market at Darrington.pdf",                 # no GT — multi-tenant
    str(BASE / "11097/20240918135005349219.pdf"),     # GT=1046359 — large center
    str(BASE / "11524/20240918140345125332.pdf"),     # GT=260000 — large center
    str(BASE / "11163/20240912133302950116.pdf"),     # GT=46444 — multi-unit flyer
    str(BASE / "10657/20240912132441386579.pdf"),     # GT=unknown
    str(BASE / "10350/20240912131941034699.pdf"),     # GT=unknown
]

# Docs that are multi-tenant / directories (run cross-model hallucination metric on these)
MULTI_TENANT = {
    "village_of_blaine.pdf",
    "Market at Darrington.pdf",
    "20240918135005349219.pdf",
    "20240918140345125332.pdf",
    "20240912133302950116.pdf",
}

GROUND_TRUTH_PATH = Path("benchmark/ground_truth.csv")
MULTI_UNIT_GT_PATH = Path("data/ground_truth_multi_unit.csv")
OUTPUT_DIR = Path("benchmark/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
@dataclass
class VisionModel:
    name: str
    backend: str          # always "openrouter"
    model_id: str
    native_pdf: bool      # send raw PDF bytes rather than rasterized pages
    cost_per_1k_input: float
    cost_per_1k_output: float
    max_pages_override: int | None = None  # hard cap below CLI --max-pages (for tight context windows)
    max_output_tokens: int = 16384        # per-model output token ceiling; 16384 is the safe default


VISION_MODELS = [
    VisionModel(
        name="Claude Haiku 4.5",
        backend="openrouter",
        model_id="anthropic/claude-haiku-4-5",
        native_pdf=False,
        cost_per_1k_input=0.00080,
        cost_per_1k_output=0.00400,
    ),
    VisionModel(
        name="Claude Sonnet 5",
        backend="openrouter",
        model_id="anthropic/claude-sonnet-5",
        native_pdf=False,
        cost_per_1k_input=0.00300,
        cost_per_1k_output=0.01500,
    ),
    VisionModel(
        name="Gemini 3.1 Pro",
        backend="openrouter",
        model_id="google/gemini-3.1-pro-preview",
        native_pdf=True,   # confirmed: accepts native PDFs same as 3.7 Flash
        cost_per_1k_input=0.00200,
        cost_per_1k_output=0.01200,
    ),
    VisionModel(
        name="Gemini 3.7 Flash",
        backend="openrouter",
        model_id="google/gemini-3.7-flash",
        native_pdf=True,   # confirmed: accepts PDFs natively, same path as 3.1 Pro
        cost_per_1k_input=0.000375,
        cost_per_1k_output=0.001875,
    ),
    VisionModel(
        name="Qwen3-VL 32B",
        backend="openrouter",
        model_id="qwen/qwen3-vl-32b-instruct",
        native_pdf=False,
        cost_per_1k_input=0.00020,
        cost_per_1k_output=0.00080,
    ),
    # GLM-4.5V: 65,536-token context, 16,384 max output → ~49K input budget.
    # Per-image tokenization rate undocumented by OpenRouter; capped at 20 pages
    # to guarantee fit even at high tokenization rates (worst-case ~2K tokens/page).
    VisionModel(
        name="GLM-4.5V",
        backend="openrouter",
        model_id="z-ai/glm-4.5v",
        native_pdf=False,
        cost_per_1k_input=0.00060,
        cost_per_1k_output=0.00180,
        max_pages_override=20,
        # max_output_tokens: uses 16384 default — model supports up to 16,384; explicit 4096 caused truncation
    ),
    VisionModel(
        name="MiMo-V2.5",
        backend="openrouter",
        model_id="xiaomi/mimo-v2.5",
        native_pdf=False,
        cost_per_1k_input=0.000119,
        cost_per_1k_output=0.000238,
    ),
    VisionModel(
        name="DeepSeek V4 Flash Vision",
        backend="openrouter",
        model_id="deepseek/deepseek-v4-flash-vision-exp",
        native_pdf=False,
        cost_per_1k_input=0.00022,
        cost_per_1k_output=0.00066,
    ),
]


# ---------------------------------------------------------------------------
# Multi-unit extraction prompt
# ---------------------------------------------------------------------------
MULTI_UNIT_PROMPT = """\
You are a commercial real estate analyst. Analyse this document — it may be an executed lease,
a leasing flyer, or a multi-tenant mall directory / tenant roster.

TASK: Identify and extract EVERY distinct unit, tenant, or leasable space visible in the
document, regardless of how it is marked — a row in a table, a numbered or colored box on a
site plan, a tenant name printed on an aerial photo, a suite description in lease text, an
inline strip label, or any other layout. Do not stop after finding the most obvious units;
scan every page systematically and include every identifiable space.

For executed leases:        one unit — the tenant's demised premises.
For leasing flyers:         each available suite listed; also include the property total if shown.
For mall directories:       every tenant row (name + SF + suite).
For ambiguous documents:    extract all distinct SF figures associated with a named tenant or suite.

Return ONLY raw JSON (no markdown, no backticks):
{
  "doc_type": "executed_lease" | "leasing_flyer" | "tenant_roster" | "unknown",
  "units": [
    {
      "tenant_name":    <string or null>,
      "square_footage": <integer, or null if no SF figure is visibly paired with this tenant>,
      "unit_number":    <string or null>,
      "floor_number":   <string or null>,
      "unit_type":      "endcap" | "inline" | "standalone" | "anchor" | "unknown",
      "confidence":     <float 0.0–1.0>,
      "source_location": <"page N, ..." or brief description of where in the document>,
      "page_number":    <integer page number where this unit was found, or null if unknown>,
      "evidence_quote": <describe exactly WHERE the SF number appears relative to the tenant name
                         — e.g. 'printed inside the unit\'s box on the site plan', 'in the same
                         table row as tenant name', 'in the lease\'s premises clause on page 2',
                         'in a box of the same color as the tenant\'s labeled unit'. If you cannot
                         point to a specific visible number for this specific unit, write
                         "low_confidence_pairing: <explain what you saw separately>">
    }
  ]
}

Rules:
- Do NOT include parking ratios, lot sizes, or land areas.
- If the same unit appears on multiple pages, include it ONCE — deduplicate by tenant_name + unit_number.
- For a total-center SF figure (e.g. "221,239 SF center"), include it as one unit with tenant_name=<property name>.
- Minimum confidence to include a unit: 0.3. Omit units you cannot place with at least 30% confidence.
- CRITICAL — exclude non-tenant sections: names appearing in a "neighbors," "nearby tenants,"
  "market area," "area retailers," "co-tenancy," or similar competitive-context section are NOT
  tenants of this property. Do NOT extract them as units regardless of how prominently they appear.
  Only extract tenants that occupy or are offered space on THIS property.
- CRITICAL — cross-reference by shared key: a unit's identifying information (tenant name) and
  its SF value may appear in two separate locations on the page, linked by a shared key rather
  than physical proximity. The shared key may be a number, letter, color, or suite code that
  appears next to BOTH the name and the value. For example: a numbered list of tenant names and
  separately numbered boxes or entries each containing an SF value — if box #3 on the site plan
  contains "1,413 SF" and the numbered tenant list shows "#3 — Great Clips", the correct pairing
  is Great Clips → 1,413 SF, even though the name and SF are in different locations. Always
  scan the entire page for such shared-key relationships before concluding that no SF is
  available for a named tenant. Only output square_footage: null if no shared-key pairing exists
  anywhere on the page for that unit.
- CRITICAL — pairing integrity: for EVERY unit you identify, square_footage must come from a
  number you can actually point to in the document. In evidence_quote, describe exactly where
  that number appears. Valid pairing evidence includes: the number appearing in the same table
  row as the tenant name; printed inside the tenant's own numbered or labeled box on a site
  plan (the number must be inside THAT box, not a nearby box); in the same labeled cell of a
  floor plan; in a lease clause that names this specific tenant. Inferring, estimating, or
  reusing a nearby value because it seems plausible is a critical failure regardless of
  document type — even if the value looks like it could fit. If you cannot point to a specific
  visible number inside or directly labeling this specific unit, output square_footage: null.
  For example, on a site plan where "Great Clips" is labeled in a box with no SF figure
  printed inside it, the correct output is square_footage: null — even if SF figures appear
  in adjacent boxes belonging to other units.
- CRITICAL — combination/rollup totals: a figure described as "Can Be Combined — up to X SF",
  "up to X SF total", "combined up to X SF", or similar rollup language describes multiple
  units combined into one contiguous space, NOT a single leasable unit. Do NOT emit a unit
  entry with that combined figure and a blank tenant name. Instead emit the individual
  component units it is composed of (each with their own SF and evidence_quote), or if the
  components are not individually listed, omit the rollup entirely.
- Named tenants with no SF figure tied to them by any of the valid pairing methods above:
  include them with square_footage: null and evidence_quote starting with
  "low_confidence_pairing:". Do NOT omit them — a name with unknown SF is useful; a
  silently dropped name is not.
"""


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
def get_openrouter_client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
REQUIRED_UNIT_FIELDS = {"tenant_name", "square_footage", "unit_number", "floor_number",
                         "unit_type", "confidence", "source_location", "evidence_quote"}
# page_number is optional (may be null) so not in REQUIRED_UNIT_FIELDS
# square_footage may be null when tenant name is known but no SF is visibly paired with it

def validate_schema(data: Any) -> tuple[bool, str]:
    """Return (is_valid, reason). Checks top-level and per-unit schema."""
    if not isinstance(data, dict):
        return False, "top-level is not a dict"
    if "units" not in data:
        return False, "missing 'units' key"
    if not isinstance(data["units"], list):
        return False, "'units' is not a list"
    if len(data["units"]) == 0:
        return False, "'units' list is empty"
    for i, u in enumerate(data["units"]):
        if not isinstance(u, dict):
            return False, f"unit[{i}] is not a dict"
        missing = REQUIRED_UNIT_FIELDS - set(u.keys())
        if missing:
            return False, f"unit[{i}] missing fields: {missing}"
        sf = u.get("square_footage")
        if sf is not None and not isinstance(sf, (int, float)):
            return False, f"unit[{i}].square_footage must be integer or null"
        if not isinstance(u.get("confidence"), (int, float)):
            return False, f"unit[{i}].confidence is not numeric"
        eq = u.get("evidence_quote", "")
        if not isinstance(eq, str) or not eq.strip():
            return False, f"unit[{i}].evidence_quote is missing or empty"
    return True, "ok"


# ---------------------------------------------------------------------------
# Extraction backends
# ---------------------------------------------------------------------------
def _calc_cost(model: VisionModel, in_tok: int, out_tok: int) -> float:
    return in_tok / 1000 * model.cost_per_1k_input + out_tok / 1000 * model.cost_per_1k_output


def _parse_units(raw: str) -> tuple[dict | None, str]:
    """Parse response JSON. Returns (parsed_dict_or_None, error_reason).
    Strips ```json ... ``` fences before parsing.
    """
    cleaned = _strip_code_block(raw)
    # _strip_code_block handles ``` and ```json openings; also strip bare language tags
    if cleaned.startswith("json\n"):
        cleaned = cleaned[5:]
    try:
        return json.loads(cleaned), ""
    except json.JSONDecodeError as e:
        # Distinguish truncation (unterminated string/array) from garbage output
        reason = f"JSONDecodeError: {e}"
        if "Unterminated string" in str(e) or "Expecting" in str(e):
            reason += " [likely truncated — check max_tokens]"
        return None, reason


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------
_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BACKOFF = [2, 5]   # seconds between attempt 1→2 and 2→3


def _call_with_retry(
    client: OpenAI,
    model_id: str,
    content: list[dict],
    max_tokens: int,
    context: str,
) -> Any:
    """Call client.chat.completions.create with up to _MAX_RETRIES attempts.
    Returns the response object on success, or an error string on final failure.
    Transient HTTP errors (429, 5xx) are retried; hard errors are not.
    """
    last_err = ""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(
                model=model_id,
                max_tokens=max_tokens,
                temperature=0,
                messages=[{"role": "user", "content": content}],
            )
        except Exception as e:
            last_err = str(e)
            status = getattr(getattr(e, "response", None), "status_code", None)
            is_transient = status in _TRANSIENT_STATUS_CODES or any(
                kw in last_err for kw in ("502", "503", "504", "timeout", "Connection")
            )
            if attempt < _MAX_RETRIES and is_transient:
                delay = _RETRY_BACKOFF[attempt - 1]
                print(f"\n      [retry {attempt}/{_MAX_RETRIES - 1} for {context} after {delay}s: {last_err[:80]}]",
                      end="", flush=True)
                time.sleep(delay)
            else:
                break
    return f"Failed after {_MAX_RETRIES} attempts — {last_err}"


_RASTER_DPI_THRESHOLDS = [
    (10,  150),   # ≤10 pages  → 150 DPI  (starting point; may step down after payload check)
    (25,  120),   # ≤25 pages  → 120 DPI
    (50,   96),   # ≤50 pages  →  96 DPI
]
# Descending ladder used by the measured-payload step-down logic.
# Starts at whatever _dpi_for_pages() returns and steps down until payload fits.
_RASTER_DPI_TIERS = [150, 120, 96, 72]
# Conservative cap for total base64 image payload (≈9 MB).
# OpenRouter's observed hard limit is ~10 MB for rasterized-image models;
# Qwen triggered a 413 at ~12.9 MB on a 3-page aerials-heavy PDF at 150 DPI.
_PAYLOAD_LIMIT_BYTES = 9_000_000
# JPEG quality used for rasterized pages. JPEG at 150 DPI passes legibility checks
# for both aerial-photo and text document types while fitting under _PAYLOAD_LIMIT_BYTES
# for all documents in the current test set (≤11 pages). The DPI step-down logic
# remains necessary for documents approaching the 50-page ceiling (~44 MB at 150 DPI JPEG).
# Results collected with JPEG encoding are NOT directly comparable to results collected
# with the previous PNG encoding — filter by the `img_encoding` result column when
# comparing runs across this boundary.
_JPEG_QUALITY = 85
_IMG_ENCODING = "jpeg"  # bump to "jpeg_v2" etc. if quality or encoding format changes


def _dpi_for_pages(n_pages: int) -> int:
    for threshold, dpi in _RASTER_DPI_THRESHOLDS:
        if n_pages <= threshold:
            return dpi
    return 96


def call_openrouter_vision(
    client: OpenAI,
    model: VisionModel,
    pdf_path: str,
    max_pages: int,
    image_save_dir: "Path | None" = None,
) -> dict:
    """Rasterize pages as JPEG (_JPEG_QUALITY) at adaptive DPI and send via OpenRouter.

    Encoding: JPEG at _JPEG_QUALITY. Verified to pass legibility checks on both aerial-photo
    and mixed text/diagram document types while reducing payload size ~6× vs PNG at the same
    DPI. DPI starts at the page-count heuristic and steps down through _RASTER_DPI_TIERS until
    the total base64 payload is under _PAYLOAD_LIMIT_BYTES (still needed for large page-count
    documents; all current test-set docs ≤11pp fit at 150 DPI JPEG without stepping).

    Results carry img_encoding=_IMG_ENCODING. Do NOT mix with pre-JPEG PNG results in
    aggregate tables — filter by img_encoding to compare like-with-like.

    If image_save_dir is provided, saves page PNGs (full-fidelity, not sent to model) as
    page_001.png etc. for UI gallery display.
    """
    import fitz
    import io as _io
    from PIL import Image as _Image

    doc = fitz.open(pdf_path)
    n_pages = min(doc.page_count, max_pages)
    doc.close()

    initial_dpi = _dpi_for_pages(n_pages)
    tiers = [d for d in _RASTER_DPI_TIERS if d <= initial_dpi]

    content: list[dict] = []
    page_pngs: list[tuple[int, bytes]] = []   # raw PNG bytes kept for gallery save
    final_dpi = initial_dpi
    total_b64_bytes = 0

    for dpi in tiers:
        content = []
        page_pngs = []
        total_b64_bytes = 0
        for pn in range(1, n_pages + 1):
            png = rasterize_page(pdf_path, pn, dpi=dpi)
            # Encode as JPEG for the API payload
            img = _Image.open(_io.BytesIO(png))
            buf = _io.BytesIO()
            img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
            jpg_bytes = buf.getvalue()
            b64 = base64.standard_b64encode(jpg_bytes).decode()
            total_b64_bytes += len(b64)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            page_pngs.append((pn, png))   # store original PNG for gallery
        final_dpi = dpi
        if total_b64_bytes <= _PAYLOAD_LIMIT_BYTES:
            break
        # payload too large; step down to next tier

    if image_save_dir is not None:
        for pn, png in page_pngs:
            (image_save_dir / f"page_{pn:03d}.png").write_bytes(png)

    content.append({"type": "text", "text": MULTI_UNIT_PROMPT})

    if final_dpi != initial_dpi:
        print(f"[DPI stepped {initial_dpi}→{final_dpi}, payload={total_b64_bytes/1e6:.1f}MB] ", end="", flush=True)

    start = time.perf_counter()
    resp = _call_with_retry(
        client, model.model_id, content, max_tokens=model.max_output_tokens,
        context=f"{model.name} / {Path(pdf_path).name}",
    )
    elapsed = time.perf_counter() - start

    if isinstance(resp, str):
        # _call_with_retry returns error string on final failure
        return _error_result(model, resp, n_pages)

    if not resp.choices:
        return _error_result(
            model,
            f"resp.choices is None/empty — model={model.model_id} pdf={Path(pdf_path).name} "
            f"resp_error={getattr(resp, 'error', None)!r}",
            n_pages,
        )

    try:
        raw = resp.choices[0].message.content
        if raw is None:
            return _error_result(
                model,
                f"choices[0].message.content is None — finish_reason={resp.choices[0].finish_reason!r} "
                f"model={model.model_id} pdf={Path(pdf_path).name}",
                n_pages,
            )
    except (IndexError, AttributeError) as e:
        return _error_result(model, f"Unexpected response structure: {e} | resp={resp!r}", n_pages)

    usage = resp.usage
    in_tok = usage.prompt_tokens if usage else 0
    out_tok = usage.completion_tokens if usage else 0
    result = _build_result(model, raw, in_tok, out_tok, elapsed, n_pages, "rasterized", pdf_path=str(pdf_path))
    result["dpi_used"] = final_dpi
    result["payload_b64_bytes"] = total_b64_bytes
    result["img_encoding"] = _IMG_ENCODING
    return result


def call_openrouter_native_pdf(
    client: OpenAI,
    model: VisionModel,
    pdf_path: str,
    max_pages: int,
) -> dict:
    """Send PDF as base64 application/pdf to models that accept it natively (Gemini)."""
    import fitz

    doc = fitz.open(pdf_path)
    n_pages = min(doc.page_count, max_pages)
    doc.close()

    pdf_bytes = Path(pdf_path).read_bytes()
    b64 = base64.standard_b64encode(pdf_bytes).decode()
    content = [
        {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{b64}"}},
        {"type": "text", "text": MULTI_UNIT_PROMPT},
    ]

    start = time.perf_counter()
    resp = _call_with_retry(
        client, model.model_id, content, max_tokens=model.max_output_tokens,
        context=f"{model.name} / {Path(pdf_path).name}",
    )
    elapsed = time.perf_counter() - start

    if isinstance(resp, str):
        return _error_result(model, resp, n_pages)

    if not resp.choices:
        return _error_result(
            model,
            f"resp.choices is None/empty — model={model.model_id} pdf={Path(pdf_path).name} "
            f"resp_error={getattr(resp, 'error', None)!r}",
            n_pages,
        )

    try:
        raw = resp.choices[0].message.content
        if raw is None:
            return _error_result(
                model,
                f"choices[0].message.content is None — finish_reason={resp.choices[0].finish_reason!r} "
                f"model={model.model_id} pdf={Path(pdf_path).name}",
                n_pages,
            )
    except (IndexError, AttributeError) as e:
        return _error_result(model, f"Unexpected response structure: {e} | resp={resp!r}", n_pages)

    usage = resp.usage
    in_tok = usage.prompt_tokens if usage else 0
    out_tok = usage.completion_tokens if usage else 0
    result = _build_result(model, raw, in_tok, out_tok, elapsed, n_pages, "native_pdf", pdf_path=str(pdf_path))
    result["img_encoding"] = "pdf"   # native PDF path; not affected by JPEG/PNG switch
    return result


def verify_citations(units: list[dict], pdf_path: str) -> list[dict]:
    """Post-process extracted units: check source_location claims against actual PDF text.

    CHECK 1 — value-presence: the cited SF value must appear in the cited page's text.
    Catches the pure-fabrication failure mode (no such value exists anywhere on the page).

    CHECK 2 — table-structure: if the citation says "table", the page must have detectable
    tabular structure. Catches phantom-table citations.

    CHECK 3 — spatial-ownership: for units whose SF value passes Check 1, verify that the
    SF value's pixel position in the PDF is nearest to the claimed unit number, not another
    unit's numbered label. Uses PyMuPDF word-level bounding boxes.
      - If the nearest unit-label is a different unit: "suspect_assignment"
      - If the nearest vs. claimed-unit distance gap is < _SPATIAL_AMBIGUITY_PX: "ambiguous_assignment"
      - If the SF appears near a different named tenant's text: "wrong_tenant_proximity"
    Check 3 runs only when:
      (a) the unit's source_location or unit_number field cites a specific unit number, AND
      (b) the cited page contains numbered-box labels (site-plan format).

    citation_status values:
      ok                    all checks passed
      unverified_citation   Check 1 or 2 failed (value absent or table phantom)
      suspect_assignment    Check 3: SF is spatially closest to a different unit number
      ambiguous_assignment  Check 3: two unit-numbers are within _SPATIAL_AMBIGUITY_PX of the SF position
      unchecked             no parseable page reference in source_location
    """
    import fitz
    import re
    import math

    _SPATIAL_AMBIGUITY_PX = 3.0  # y-distance gap below which two candidates are "too close to call"

    # Open once, cache page text + word-level positions
    _page_texts: dict[int, str] = {}
    _page_words: dict[int, list] = {}   # page_num → list of (x0,y0,x1,y1,word)
    try:
        doc = fitz.open(pdf_path)
        for i in range(doc.page_count):
            pg = doc[i]
            _page_texts[i + 1] = pg.get_text()
            _page_words[i + 1] = [(w[0], w[1], w[2], w[3], w[4]) for w in pg.get_text("words")]
        doc.close()
    except Exception:
        for u in units:
            u["citation_status"] = "unchecked"
            u["citation_note"] = "could not open PDF for text extraction"
        return units

    _page_re = re.compile(r'\bpage\s+(\d+)\b', re.IGNORECASE)
    _table_re = re.compile(r'\btable\b', re.IGNORECASE)
    _unit_num_re = re.compile(r'^(\d{1,2})$')
    _sf_digit_re = re.compile(r'^[\d,]+$')

    def _sf_as_text(sf) -> list[str]:
        if sf is None:
            return []
        n = int(sf)
        forms = [str(n)]
        if n >= 1000:
            forms.append(f"{n:,}")
        return forms

    def _looks_tabular(text: str) -> bool:
        if '\t' in text or '|' in text:
            return True
        return sum(1 for line in text.splitlines() if len(re.findall(r'\b[\d,]+\b', line)) >= 2) >= 3

    def _find_sf_positions(words: list, sf: int, y_min: float = 0, y_max: float = 1e9):
        """Return [(x_center, y_center)] for all occurrences of sf on the page within y bounds."""
        targets = {str(sf), f"{sf:,}"}
        hits = []
        for x0, y0, x1, y1, w in words:
            if w.strip().replace(',', '').replace('.', '') == str(sf) or w.strip() in targets:
                if y_min <= y0 <= y_max:
                    hits.append(((x0 + x1) / 2, (y0 + y1) / 2))
        return hits

    def _find_unit_labels(words: list, y_min: float = 0, y_max: float = 1e9) -> dict[int, tuple[float,float]]:
        """Return {unit_num: (x_center, y_center)} for numbered box labels within y bounds.
        Takes the first occurrence of each number (avoids picking up legend rows)."""
        labels: dict[int, tuple[float,float]] = {}
        for x0, y0, x1, y1, w in words:
            m = _unit_num_re.match(w.strip())
            if m and y_min <= y0 <= y_max:
                n = int(m.group(1))
                if 1 <= n <= 99 and n not in labels:
                    labels[n] = ((x0 + x1) / 2, (y0 + y1) / 2)
        return labels

    def _dist_2d(a: tuple[float,float], b: tuple[float,float]) -> float:
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def _plan_axis(unit_labels: dict[int, tuple[float,float]]) -> str:
        """Detect whether the site plan's numbered boxes run horizontally or vertically.
        Returns 'x' (horizontal strip) or 'y' (vertical strip)."""
        if len(unit_labels) < 2:
            return 'y'
        positions = list(unit_labels.values())
        x_span = max(p[0] for p in positions) - min(p[0] for p in positions)
        y_span = max(p[1] for p in positions) - min(p[1] for p in positions)
        return 'x' if x_span > y_span else 'y'

    def _check_spatial_ownership(unit: dict, sf: int, cited_page: int) -> list[str]:
        """Run Check 3 using box-containment, not nearest-label distance.

        SF values are printed INSIDE their box: they appear between label N (top of box)
        and label N+1 (top of the next box). Nearest-label distance is always wrong because
        the SF sits below its own label and above the next one. Containment with a small
        tolerance handles values printed at box edges.

        Tries the cited page first, then other pages, stopping at the first page that has
        BOTH the SF value and the claimed unit label in the plan area.
        """
        # _SPATIAL_AMBIGUITY_PX is used as the containment tolerance (px slack at box edges)
        _CONTAINMENT_TOLERANCE_PX = 3.0

        claimed_unit_str = str(unit.get("unit_number") or "").strip()
        if not claimed_unit_str or not claimed_unit_str.isdigit():
            return []
        claimed_unit = int(claimed_unit_str)

        all_pages = list(_page_words.keys())
        pages_to_try = [cited_page] + [p for p in all_pages if p != cited_page]

        for page_num in pages_to_try:
            words = _page_words.get(page_num, [])
            if not words:
                continue

            ys = [w[1] for w in words]
            y_page_min, y_page_max = min(ys), max(ys)
            plan_y_min = y_page_min + 0.10 * (y_page_max - y_page_min)
            plan_y_max = y_page_min + 0.90 * (y_page_max - y_page_min)

            sf_positions = _find_sf_positions(words, sf, y_min=plan_y_min, y_max=plan_y_max)
            unit_labels = _find_unit_labels(words, y_min=plan_y_min, y_max=plan_y_max)

            if not sf_positions or claimed_unit not in unit_labels:
                continue

            sf_pos = sf_positions[0]
            axis = _plan_axis(unit_labels)

            # Get the coordinate of the SF value and unit labels along the strip axis
            sf_coord   = sf_pos[0] if axis == 'x' else sf_pos[1]
            label_coord = lambda n: unit_labels[n][0] if axis == 'x' else unit_labels[n][1]

            # Sort labels along the strip axis to find the next label after claimed_unit
            strip_sorted = sorted(
                [(n, label_coord(n)) for n in unit_labels if 1 <= n <= 99],
                key=lambda kv: kv[1]
            )
            claimed_coord = label_coord(claimed_unit)
            claimed_idx = next((i for i, (n, _) in enumerate(strip_sorted) if n == claimed_unit), None)
            if claimed_idx is None:
                continue

            # Box boundary: from claimed unit's label coord to next unit's label coord
            next_coord = strip_sorted[claimed_idx + 1][1] if claimed_idx + 1 < len(strip_sorted) else float('inf')

            # Containment with tolerance: sf_coord should be in [claimed_coord - tol, next_coord + tol]
            in_box = (claimed_coord - _CONTAINMENT_TOLERANCE_PX
                      <= sf_coord
                      <= next_coord + _CONTAINMENT_TOLERANCE_PX)

            if not in_box:
                # Find which box it actually falls in
                actual_unit = None
                for i, (n, coord) in enumerate(strip_sorted):
                    next_c = strip_sorted[i + 1][1] if i + 1 < len(strip_sorted) else float('inf')
                    if coord - _CONTAINMENT_TOLERANCE_PX <= sf_coord <= next_c + _CONTAINMENT_TOLERANCE_PX:
                        actual_unit = n
                        break
                if actual_unit and actual_unit != claimed_unit:
                    return [
                        f"suspect_assignment: SF {sf:,} falls in box for unit {actual_unit} "
                        f"(coord={sf_coord:.1f}; unit {actual_unit} range [{label_coord(actual_unit):.1f}, …]) "
                        f"not claimed unit {claimed_unit} (range [{claimed_coord:.1f}, {next_coord:.1f}]); "
                        f"checked page {page_num} axis={axis}"
                    ]
            return []  # contained in claimed unit's box — ok

        # Could not verify on any page (SF or label not found at word level)
        return []

    for unit in units:
        src = (unit.get("source_location") or unit.get("evidence_quote") or "").strip()
        sf = unit.get("square_footage")

        if not src:
            unit["citation_status"] = "unchecked"
            unit["citation_note"] = "no source_location to verify"
            continue

        page_match = _page_re.search(src)
        if not page_match:
            unit["citation_status"] = "unchecked"
            unit["citation_note"] = "source_location has no parseable page number"
            continue

        cited_page = int(page_match.group(1))
        page_text = _page_texts.get(cited_page, "")

        failures: list[str] = []

        # Check 1: SF value must appear on the cited page text
        if sf is not None:
            sf_forms = _sf_as_text(sf)
            if not any(form in page_text for form in sf_forms):
                failures.append(
                    f"SF {sf} ({'/'.join(sf_forms)}) not found in page {cited_page} text"
                )

        # Check 2: table-structure claim must be verifiable
        if _table_re.search(src) and not _looks_tabular(page_text):
            failures.append(
                f"citation claims 'table' on page {cited_page} but page text shows no tabular structure"
            )

        # Check 3: spatial ownership (only if Checks 1+2 passed)
        if not failures and sf is not None:
            failures.extend(_check_spatial_ownership(unit, int(sf), cited_page))

        if failures:
            # Determine final status
            has_spatial = any("assignment" in f for f in failures)
            has_phantom = any("not found" in f or "table" in f for f in failures)
            if has_phantom:
                unit["citation_status"] = "unverified_citation"
            elif any("ambiguous_assignment" in f for f in failures):
                unit["citation_status"] = "ambiguous_assignment"
            elif any("suspect_assignment" in f for f in failures):
                unit["citation_status"] = "suspect_assignment"
            else:
                unit["citation_status"] = "unverified_citation"
            unit["citation_note"] = "; ".join(failures)
        else:
            unit["citation_status"] = "ok"
            unit["citation_note"] = ""

    return units


def _build_result(
    model: VisionModel,
    raw: str,
    in_tok: int,
    out_tok: int,
    elapsed: float,
    n_pages: int,
    input_method: str,
    pdf_path: str = "",
) -> dict:
    parsed, parse_err = _parse_units(raw)
    if parsed is None:
        valid, reason = False, parse_err
        units = []
    else:
        valid, reason = validate_schema(parsed)
        units = parsed.get("units", []) if valid else []

    if units and pdf_path:
        verify_citations(units, pdf_path)

    return {
        "model_name": model.name,
        "model_id": model.model_id,
        "input_method": input_method,
        "n_pages": n_pages,
        "json_valid": valid,
        "schema_error": "" if valid else reason,
        "units": units,
        "doc_type": (parsed or {}).get("doc_type", "unknown"),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": _calc_cost(model, in_tok, out_tok),
        "latency_sec": round(elapsed, 2),
        "raw_response": raw,
    }


def _error_result(model: VisionModel, msg: str, n_pages: int) -> dict:
    return {
        "model_name": model.name,
        "model_id": model.model_id,
        "input_method": "error",
        "n_pages": n_pages,
        "json_valid": False,
        "schema_error": msg[:300],
        "units": [],
        "doc_type": "unknown",
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "latency_sec": 0.0,
        "raw_response": "",
    }


# ---------------------------------------------------------------------------
# Hallucination / duplication analysis
# ---------------------------------------------------------------------------
def _normalize_name(name: str | None) -> str:
    if not name:
        return ""
    return name.lower().strip().replace(",", "").replace(".", "")


def _units_match(a: dict, b: dict, sf_tol: float = 0.15) -> bool:
    """Two units match if their tenant names overlap AND SF is within tolerance."""
    na, nb = _normalize_name(a.get("tenant_name")), _normalize_name(b.get("tenant_name"))
    if na and nb:
        # substring match (covers "Chipotle Mexican Grill" vs "Chipotle")
        name_ok = na in nb or nb in na
    else:
        name_ok = False

    sfa, sfb = a.get("square_footage", 0), b.get("square_footage", 0)
    if sfa and sfb:
        sf_ok = abs(sfa - sfb) / max(sfa, sfb) <= sf_tol
    else:
        sf_ok = True  # can't compare

    return name_ok and sf_ok


def hallucination_report(model_results: dict[str, list[dict]]) -> dict:
    """
    Cross-model diff for a single document.
    model_results: {model_name: [unit, ...]}
    Returns per-model disagreement rates and within-model duplicates.
    """
    model_names = list(model_results.keys())
    per_model: dict[str, dict] = {}

    for name, units in model_results.items():
        # Within-model duplicates: same tenant appears 2+ times
        seen: list[dict] = []
        dupes = 0
        for u in units:
            if any(_units_match(u, s) for s in seen):
                dupes += 1
            else:
                seen.append(u)
        per_model[name] = {"total_units": len(units), "within_model_dupes": dupes}

    # Cross-model: for each model, count units no other model found
    for name, units in model_results.items():
        others = [u for m, us in model_results.items() if m != name for u in us]
        unique = sum(
            1 for u in units
            if not any(_units_match(u, o) for o in others)
        )
        per_model[name]["unique_to_this_model"] = unique
        total = per_model[name]["total_units"]
        per_model[name]["unique_rate"] = round(unique / total, 3) if total else 0.0

    return per_model


# ---------------------------------------------------------------------------
# Ground truth — single-value (original benchmark) and per-unit (new)
# ---------------------------------------------------------------------------
def load_ground_truth() -> dict[str, int | None]:
    """Single correct_sf per PDF from benchmark/ground_truth.csv."""
    if not GROUND_TRUTH_PATH.exists():
        return {}
    gt: dict[str, int | None] = {}
    with GROUND_TRUTH_PATH.open() as fh:
        for row in csv.DictReader(fh):
            val = row.get("correct_sf", "").strip()
            gt[row["pdf"]] = int(val) if val else None
    return gt


def load_multi_unit_gt() -> dict[str, list[dict]]:
    """Per-unit ground truth from data/ground_truth_multi_unit.csv.
    Returns {pdf_basename: [{"tenant_name", "unit_number", "floor_number", "unit_type", "sqft"}, ...]}.
    """
    if not MULTI_UNIT_GT_PATH.exists():
        return {}
    gt: dict[str, list[dict]] = defaultdict(list)
    with MULTI_UNIT_GT_PATH.open() as fh:
        for row in csv.DictReader(fh):
            sqft_str = row.get("sqft", "").strip()
            gt[row["doc_id"].strip()].append({
                "tenant_name":  row.get("tenant_name", "").strip() or None,
                "unit_number":  row.get("unit_number", "").strip() or None,
                "floor_number": row.get("floor_number", "").strip() or None,
                "unit_type":    row.get("unit_type", "").strip() or None,
                "sqft":         int(sqft_str) if sqft_str else None,
            })
    return dict(gt)


def _gt_unit_match(model_unit: dict, gt_unit: dict) -> bool:
    """True if model_unit refers to the same physical unit as gt_unit.
    Requires tenant name overlap; if both sides have a unit_number they must agree.
    """
    mn = _normalize_name(model_unit.get("tenant_name"))
    gn = _normalize_name(gt_unit.get("tenant_name"))
    if not mn or not gn:
        return False
    if mn not in gn and gn not in mn:
        return False
    mu_num = (model_unit.get("unit_number") or "").strip().lower()
    gt_num = (gt_unit.get("unit_number") or "").strip().lower()
    if mu_num and gt_num and mu_num != gt_num:
        return False
    return True


def score_against_gt(
    model_units: list[dict],
    gt_units: list[dict],
    sf_tol: float = 0.10,
) -> dict:
    """Diff model_units against gt_units for one document.

    Returns:
      gt_count       — total GT units for this doc
      units_correct  — GT units matched AND SF within ±10%
      units_missed   — GT units not matched by any model unit
      units_invented — model units not matched to any GT unit
      units_duped    — model units that matched a GT unit already claimed by an earlier model unit
    """
    gt_claimed = [False] * len(gt_units)
    model_to_gt: list[int | None] = []

    for mu in model_units:
        matched_idx = None
        for gi, gu in enumerate(gt_units):
            if _gt_unit_match(mu, gu):
                matched_idx = gi
                break
        model_to_gt.append(matched_idx)

    correct = 0
    duped = 0
    invented = 0
    for mu, gi in zip(model_units, model_to_gt):
        if gi is None:
            invented += 1
        elif gt_claimed[gi]:
            duped += 1
        else:
            gt_claimed[gi] = True
            gt_sf = gt_units[gi].get("sqft")
            mu_sf = mu.get("square_footage", 0)
            if gt_sf and mu_sf and abs(mu_sf - gt_sf) / gt_sf <= sf_tol:
                correct += 1

    missed = sum(1 for c in gt_claimed if not c)
    return {
        "gt_count":      len(gt_units),
        "units_correct": correct,
        "units_missed":  missed,
        "units_invented": invented,
        "units_duped":   duped,
    }


_EMPTY_GT_SCORES: dict = {
    "gt_count": None, "units_correct": None,
    "units_missed": None, "units_invented": None, "units_duped": None,
}


def accuracy_score(units: list[dict], correct_sf: int | None, tolerance: float = 0.10) -> bool | None:
    """Fallback: True if any unit is within ±10% of the single-value correct_sf."""
    if correct_sf is None:
        return None
    if not units:
        return False
    for u in units:
        sf = u.get("square_footage", 0)
        if sf and abs(sf - correct_sf) / correct_sf <= tolerance:
            return True
    return False


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------
def run_pdf(
    pdf_path: str,
    models: list[VisionModel],
    or_client: OpenAI,
    max_pages: int,
    multi_gt: dict[str, list[dict]],
) -> list[dict]:
    path = Path(pdf_path)
    gt_units_for_doc = multi_gt.get(path.name, [])
    has_gt = bool(gt_units_for_doc)
    print(f"\n  PDF: {path.name} ({path.stat().st_size // 1024} KB)" +
          (f" [{len(gt_units_for_doc)} GT units]" if has_gt else ""))

    rows = []
    for model in models:
        effective_max = max_pages
        if model.max_pages_override is not None:
            effective_max = min(max_pages, model.max_pages_override)
        print(f"    [{model.name}]...", end=" ", flush=True)
        if model.native_pdf:
            result = call_openrouter_native_pdf(or_client, model, pdf_path, effective_max)
        else:
            result = call_openrouter_vision(or_client, model, pdf_path, effective_max)

        result["pdf"] = path.name
        result["pdf_path"] = pdf_path

        if has_gt:
            gt_scores = score_against_gt(result.get("units", []), gt_units_for_doc)
        else:
            gt_scores = dict(_EMPTY_GT_SCORES)
        result.update(gt_scores)

        n_units = len(result["units"])
        valid_str = "✓" if result["json_valid"] else "✗"
        gt_str = (f" | GT: {gt_scores['units_correct']}/{gt_scores['gt_count']} correct"
                  f" miss={gt_scores['units_missed']} inv={gt_scores['units_invented']}"
                  if has_gt else "")
        print(
            f"{valid_str} {n_units} units | ${result['cost_usd']:.5f} | "
            f"{result['latency_sec']}s{gt_str}"
            + (f" | {result['schema_error']}" if result.get("schema_error") else "")
        )
        rows.append(result)

    return rows


_UNIT_FIELDS = ["tenant_name", "square_footage", "unit_number",
                "floor_number", "unit_type", "confidence", "source_location",
                "citation_status", "citation_note"]
_GT_SCORE_FIELDS = ["gt_count", "units_correct", "units_missed", "units_invented", "units_duped"]


def write_results(rows: list[dict], run_id: str) -> Path:
    """Flatten units list and write to CSV.
    GT score columns (gt_count, units_correct, …) appear once per doc/model row,
    repeated across every unit row for that combination so the CSV is filterable.
    """
    flat_rows = []
    for r in rows:
        base = {k: v for k, v in r.items() if k not in ("units", "raw_response")}
        units = r.get("units", [])
        empty_unit = {"unit_index": None, **{k: None for k in _UNIT_FIELDS}}
        if not units:
            flat_rows.append({**base, **empty_unit})
        else:
            for i, u in enumerate(units):
                flat_rows.append({**base, "unit_index": i,
                                  **{k: u.get(k) for k in _UNIT_FIELDS}})

    out = OUTPUT_DIR / f"vision_benchmark_{run_id}.csv"
    if flat_rows:
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)
    return out


def print_summary(rows: list[dict]) -> None:
    sv_gt = load_ground_truth()   # single-value fallback

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model_name"]].append(r)

    by_pdf: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_pdf[r["pdf"]][r["model_name"]] = r.get("units", [])

    W = 125
    print(f"\n{'='*W}")
    print("VISION BENCHMARK SUMMARY")
    print(f"{'='*W}")
    hdr = (f"{'Model':<22} {'Input':>12} {'Valid%':>7} {'Units/doc':>10} "
           f"{'UnitAcc%':>9} {'Miss%':>7} {'Invent%':>8} {'$/doc':>9} {'Lat(s)':>8}")
    print(hdr)
    print(f"{'-'*W}")

    for model_name, model_rows in sorted(by_model.items()):
        n = len(model_rows)
        valid_pct = 100 * sum(1 for r in model_rows if r["json_valid"]) / n if n else 0
        all_units = [u for r in model_rows for u in r.get("units", [])]
        avg_units = len(all_units) / n if n else 0
        avg_cost = sum(r["cost_usd"] for r in model_rows) / n if n else 0
        avg_lat  = sum(r["latency_sec"] for r in model_rows) / n if n else 0
        backend  = (model_rows[0].get("input_method") or "").replace("_", " ")

        # Per-unit GT accuracy (preferred when multi_unit GT is present)
        mu_rows = [r for r in model_rows if r.get("gt_count") is not None]
        if mu_rows:
            total_gt  = sum(r["gt_count"]      for r in mu_rows)
            total_cor = sum(r["units_correct"]  for r in mu_rows)
            total_mis = sum(r["units_missed"]   for r in mu_rows)
            total_inv = sum(r["units_invented"] for r in mu_rows)
            acc_str   = f"{100*total_cor/total_gt:.1f}%" if total_gt else "N/A"
            miss_str  = f"{100*total_mis/total_gt:.1f}%" if total_gt else "N/A"
            inv_denom = sum(len(r.get("units", [])) for r in mu_rows)
            inv_str   = f"{100*total_inv/inv_denom:.1f}%" if inv_denom else "N/A"
        else:
            # Fall back to single-value GT
            sv_rows = [(r, sv_gt.get(r["pdf"])) for r in model_rows if sv_gt.get(r["pdf"]) is not None]
            if sv_rows:
                correct = sum(1 for r, g in sv_rows if accuracy_score(r.get("units", []), g) is True)
                acc_str = f"{correct}/{len(sv_rows)}"
            else:
                acc_str = "N/A"
            miss_str = "N/A"
            inv_str  = "N/A"

        print(f"{model_name:<22} {backend:>12} {valid_pct:>6.1f}% {avg_units:>10.1f} "
              f"{acc_str:>9} {miss_str:>7} {inv_str:>8} ${avg_cost:>7.5f} {avg_lat:>7.2f}s")

    # -----------------------------------------------------------------------
    # GT-based hallucination table (docs with per-unit ground truth)
    # -----------------------------------------------------------------------
    gt_docs = {pdf for pdf, model_units in by_pdf.items()
               if any(r.get("gt_count") for r in rows if r["pdf"] == pdf)}

    if gt_docs:
        print(f"\n{'='*W}")
        print("HALLUCINATION vs GROUND TRUTH (docs with per-unit GT)")
        print(f"{'='*W}")
        print(f"{'Document':<35} {'Model':<22} {'GT':>4} {'Correct':>8} {'Missed':>7} {'Invented':>9} {'Duped':>6}")
        print(f"{'-'*W}")
        for pdf_name in sorted(gt_docs):
            first = True
            for r in sorted(rows, key=lambda x: x["model_name"]):
                if r["pdf"] != pdf_name or r.get("gt_count") is None:
                    continue
                doc_label = pdf_name[:34] if first else ""
                first = False
                print(
                    f"{doc_label:<35} {r['model_name']:<22} {r['gt_count']:>4} "
                    f"{r['units_correct']:>8} {r['units_missed']:>7} "
                    f"{r['units_invented']:>9} {r['units_duped']:>6}"
                )

    # -----------------------------------------------------------------------
    # Cross-model hallucination (multi-tenant docs without per-unit GT)
    # -----------------------------------------------------------------------
    no_gt_multi = {pdf for pdf in MULTI_TENANT if pdf not in gt_docs}
    cross_model_rows = [(pdf, by_pdf[pdf]) for pdf in no_gt_multi if len(by_pdf[pdf]) >= 2]
    if cross_model_rows:
        print(f"\n{'='*W}")
        print("CROSS-MODEL HALLUCINATION (multi-tenant docs without GT)")
        print(f"{'='*W}")
        print(f"{'Document':<35} {'Model':<22} {'Total':>6} {'Dupes':>6} {'Unique':>7} {'UniqueRate':>11}")
        print(f"{'-'*W}")
        for pdf_name, model_units in sorted(cross_model_rows):
            report = hallucination_report(dict(model_units))
            first = True
            for mname, stats in sorted(report.items()):
                doc_label = pdf_name[:34] if first else ""
                first = False
                print(
                    f"{doc_label:<35} {mname:<22} {stats['total_units']:>6} "
                    f"{stats['within_model_dupes']:>6} {stats['unique_to_this_model']:>7} "
                    f"{stats['unique_rate']:>10.1%}"
                )

    print(f"\n{'='*W}")
    print(f"Total spend this run: ${sum(r['cost_usd'] for r in rows):.4f}")
    print(f"{'='*W}\n")

    _print_recommendation(by_model, sv_gt)


def _print_recommendation(by_model: dict[str, list[dict]], gt: dict) -> None:
    """Print a brief production recommendation based on results."""
    # Find models with >80% JSON validity
    qualified = []
    for name, model_rows in by_model.items():
        n = len(model_rows)
        valid_pct = sum(1 for r in model_rows if r["json_valid"]) / n if n else 0
        acc_rows = [(r, gt.get(r["pdf"])) for r in model_rows if gt.get(r["pdf"]) is not None]
        acc = sum(1 for r, g in acc_rows if accuracy_score(r.get("units", []), g)) / len(acc_rows) if acc_rows else 0
        avg_cost = sum(r["cost_usd"] for r in model_rows) / n if n else 0
        qualified.append((name, valid_pct, acc, avg_cost))

    qualified.sort(key=lambda x: (-x[2], -x[1], x[3]))

    print("RECOMMENDATION")
    print("-" * 80)
    if not qualified:
        print("No models produced results to compare.")
        return

    best = qualified[0]
    cheapest = min(qualified, key=lambda x: x[3])

    print(
        f"Top accuracy: {best[0]} (validity={best[1]:.0%}, accuracy={best[2]:.0%}, "
        f"${best[3]:.5f}/doc)."
    )
    if cheapest[0] != best[0]:
        ratio = best[3] / cheapest[3] if cheapest[3] > 0 else float("inf")
        print(
            f"Cheapest viable option: {cheapest[0]} at ${cheapest[3]:.5f}/doc "
            f"({ratio:.0f}x less expensive than {best[0]})."
        )
        print(
            f"Consider a tiered approach: {cheapest[0]} for volume processing, "
            f"escalate to {best[0]} when confidence < 0.7 or units == 0."
        )
    else:
        print(f"{best[0]} is both highest-accuracy and lowest-cost — use it directly.")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Vision model benchmark — single-pass PDF extraction")
    parser.add_argument("--models",    nargs="+", help="Name keywords to filter models (e.g. haiku sonnet gemini)")
    parser.add_argument("--pdf",       help="Run on a single PDF instead of the test set")
    parser.add_argument("--max-pages", type=int, default=50, help="Max pages to send per document (default 50)")
    args = parser.parse_args()

    models = VISION_MODELS
    if args.models:
        kws = [k.lower() for k in args.models]
        models = [m for m in VISION_MODELS if any(kw in m.name.lower() or kw in m.model_id.lower() for kw in kws)]
        if not models:
            print(f"No models matched: {args.models}")
            sys.exit(1)

    pdfs = [args.pdf] if args.pdf else TEST_PDFS
    pdfs = [p for p in pdfs if Path(p).exists()]
    if not pdfs:
        print("No PDFs found — check paths or run from project root.")
        sys.exit(1)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    or_client = get_openrouter_client()
    multi_gt = load_multi_unit_gt()

    print(f"\nVision Benchmark: {run_id}")
    print(f"Models:    {', '.join(m.name for m in models)}")
    print(f"PDFs:      {len(pdfs)}")
    print(f"Max pages: {args.max_pages}")
    if multi_gt:
        print(f"GT docs:   {len(multi_gt)} ({sum(len(v) for v in multi_gt.values())} units)")
    else:
        print(f"GT docs:   none — place data/ground_truth_multi_unit.csv to enable per-unit scoring")
    print("-" * 60)

    all_rows: list[dict] = []
    for pdf in pdfs:
        rows = run_pdf(pdf, models, or_client, args.max_pages, multi_gt)
        all_rows.extend(rows)

    out = write_results(all_rows, run_id)
    print_summary(all_rows)
    print(f"Full results → {out}")


if __name__ == "__main__":
    main()
