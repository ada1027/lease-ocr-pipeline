"""
Isolation test: identify which of the two prompt instructions added to
MULTI_UNIT_PROMPT causes regressions on document 10782.

Variants:
  baseline  — MULTI_UNIT_PROMPT with BOTH new rules removed
  nbrs_only — MULTI_UNIT_PROMPT with only the neighbors-exclusion rule
  xref_only — MULTI_UNIT_PROMPT with only the cross-reference rule
  both      — MULTI_UNIT_PROMPT unmodified (what the app actually sends)

IMPORTANT: prompts are derived by removing named sections from the live
MULTI_UNIT_PROMPT constant. The script never constructs its own copy of
any prompt text. The "both" variant must hash-match the live constant;
the script aborts if it does not.
"""

import hashlib
import io
import os
import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.vision_run import (
    MULTI_UNIT_PROMPT,
    VISION_MODELS,
    get_openrouter_client,
    _call_with_retry,
    _build_result,
    rasterize_page,
    _RASTER_DPI_TIERS,
    _PAYLOAD_LIMIT_BYTES,
    _JPEG_QUALITY,
    _IMG_ENCODING,
    _dpi_for_pages,
)
from PIL import Image as _Image

PDF_PATH = "data/SiteFiles_20260625_part01/SiteFiles/10782/20240912132626418342.pdf"
MAX_PAGES = 20

GT_AVAILABLE = {2: 2500, 3: 2844, 5: 1413, 7: 1486, 8: 1365, 9: 6474}
NEIGHBOR_KEYWORDS = {
    "best market", "modell", "rite aid", "stop & shop", "starbucks",
    "mcdonald", "chase bank", "td bank", "cvs", "walgreens", "shoprite",
}

# ---------------------------------------------------------------------------
# Named sections to remove from MULTI_UNIT_PROMPT when building variants.
# Each sentinel must appear EXACTLY ONCE in MULTI_UNIT_PROMPT; verified below.
# ---------------------------------------------------------------------------

_SECTION_NEIGHBORS = (
    "- CRITICAL — exclude non-tenant sections: names appearing in a \"neighbors,\" \"nearby tenants,\"\n"
    "  \"market area,\" \"area retailers,\" \"co-tenancy,\" or similar competitive-context section are NOT\n"
    "  tenants of this property. Do NOT extract them as units regardless of how prominently they appear.\n"
    "  Only extract tenants that occupy or are offered space on THIS property.\n"
)

_SECTION_XREF = (
    "- CRITICAL — cross-reference by shared key: a unit's identifying information (tenant name) and\n"
    "  its SF value may appear in two separate locations on the page, linked by a shared key rather\n"
    "  than physical proximity. The shared key may be a number, letter, color, or suite code that\n"
    "  appears next to BOTH the name and the value. For example: a numbered list of tenant names and\n"
    "  separately numbered boxes or entries each containing an SF value — if box #3 on the site plan\n"
    "  contains \"1,413 SF\" and the numbered tenant list shows \"#3 — Great Clips\", the correct pairing\n"
    "  is Great Clips → 1,413 SF, even though the name and SF are in different locations. Always\n"
    "  scan the entire page for such shared-key relationships before concluding that no SF is\n"
    "  available for a named tenant. Only output square_footage: null if no shared-key pairing exists\n"
    "  anywhere on the page for that unit.\n"
)


def _remove(prompt: str, section: str, label: str) -> str:
    """Remove section from prompt; abort if not found exactly once."""
    count = prompt.count(section)
    if count != 1:
        raise AssertionError(
            f"Section '{label}' appears {count} times in MULTI_UNIT_PROMPT "
            f"(expected exactly 1). Script is out of sync with vision_run.py — "
            f"update _SECTION_{label.upper()} to match the current prompt text."
        )
    return prompt.replace(section, "")


def _build_variants() -> dict[str, str]:
    both = MULTI_UNIT_PROMPT          # unmodified — what the app sends
    no_nbrs = _remove(both, _SECTION_NEIGHBORS, "NEIGHBORS")
    no_xref = _remove(both, _SECTION_XREF, "XREF")
    base = _remove(no_nbrs, _SECTION_XREF, "XREF")
    return {
        "baseline":  base,
        "nbrs_only": _remove(both, _SECTION_XREF, "XREF"),
        "xref_only": _remove(both, _SECTION_NEIGHBORS, "NEIGHBORS"),
        "both":      both,
    }


# ---------------------------------------------------------------------------
# Integrity guard — must pass before any API calls are made
# ---------------------------------------------------------------------------

def _assert_integrity(variants: dict[str, str]) -> None:
    live_hash = hashlib.sha256(MULTI_UNIT_PROMPT.encode()).hexdigest()
    both_hash = hashlib.sha256(variants["both"].encode()).hexdigest()

    if both_hash != live_hash:
        raise AssertionError(
            f"INTEGRITY FAILURE: variants['both'] hash {both_hash[:16]} != "
            f"MULTI_UNIT_PROMPT hash {live_hash[:16]}. "
            f"The 'both' variant must be byte-identical to the live prompt."
        )

    # Verify sections are present in "both" but absent from "baseline"
    for section, label in [(_SECTION_NEIGHBORS, "NEIGHBORS"), (_SECTION_XREF, "XREF")]:
        if section not in variants["both"]:
            raise AssertionError(f"Section {label} not found in 'both' variant.")
        if section in variants["baseline"]:
            raise AssertionError(f"Section {label} still present in 'baseline' variant.")

    # Print confirmation
    print("Integrity check PASSED")
    for name, prompt in variants.items():
        h = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        has_nbrs = _SECTION_NEIGHBORS in prompt
        has_xref = _SECTION_XREF in prompt
        print(f"  [{name:9}]  SHA256={h}  len={len(prompt):5}  "
              f"nbrs={'Y' if has_nbrs else 'N'}  xref={'Y' if has_xref else 'N'}"
              f"{'  ← live prompt (app)' if name == 'both' else ''}")
    print()


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def _run_native_pdf(client, model, pdf_path: str, prompt: str) -> dict:
    import fitz
    doc = fitz.open(pdf_path)
    n_pages = min(doc.page_count, MAX_PAGES)
    doc.close()
    pdf_bytes = Path(pdf_path).read_bytes()
    b64 = base64.standard_b64encode(pdf_bytes).decode()
    content = [
        {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]
    start = time.perf_counter()
    resp = _call_with_retry(client, model.model_id, content, model.max_output_tokens, model.name)
    elapsed = time.perf_counter() - start
    if isinstance(resp, str):
        return {"error": resp, "units": []}
    usage = resp.usage
    return _build_result(
        model,
        resp.choices[0].message.content or "",
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
        elapsed, n_pages, "native_pdf", pdf_path=pdf_path,
    )


def _run_rasterized(client, model, pdf_path: str, prompt: str) -> dict:
    import fitz
    doc = fitz.open(pdf_path)
    n_pages = min(doc.page_count, MAX_PAGES)
    doc.close()
    initial_dpi = _dpi_for_pages(n_pages)
    tiers = [d for d in _RASTER_DPI_TIERS if d <= initial_dpi]
    content: list[dict] = []
    total_b64 = 0
    final_dpi = initial_dpi
    for dpi in tiers:
        content = []
        total_b64 = 0
        for pn in range(1, n_pages + 1):
            png = rasterize_page(pdf_path, pn, dpi=dpi)
            img = _Image.open(io.BytesIO(png))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
            b64 = base64.standard_b64encode(buf.getvalue()).decode()
            total_b64 += len(b64)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        final_dpi = dpi
        if total_b64 <= _PAYLOAD_LIMIT_BYTES:
            break
    content.append({"type": "text", "text": prompt})
    start = time.perf_counter()
    resp = _call_with_retry(client, model.model_id, content, model.max_output_tokens, model.name)
    elapsed = time.perf_counter() - start
    if isinstance(resp, str):
        return {"error": resp, "units": []}
    usage = resp.usage
    return _build_result(
        model,
        resp.choices[0].message.content or "",
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
        elapsed, n_pages, "rasterized", pdf_path=pdf_path,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(result: dict, model_label: str, variant_label: str) -> None:
    units = result.get("units", [])
    if not units and "error" in result:
        print(f"  ERROR: {result['error'][:120]}")
        return

    nbr_hits = [
        u.get("tenant_name", "")
        for u in units
        if any(kw in (u.get("tenant_name") or "").lower() for kw in NEIGHBOR_KEYWORDS)
    ]

    # Non-GT SF: any SF value that isn't one of the 6 known available-unit values
    non_gt_sf = [
        (u.get("tenant_name", "?"), u.get("square_footage"), u.get("citation_status", ""))
        for u in units
        if u.get("square_footage") and u.get("square_footage") not in GT_AVAILABLE.values()
    ]

    avail_found: dict[int, str] = {}
    for u in units:
        sf = u.get("square_footage")
        un = str(u.get("unit_number") or "")
        if sf in GT_AVAILABLE.values():
            avail_found[sf] = un

    correct_c = sum(
        1 for sf, un in avail_found.items()
        if un in [str(k) for k, v in GT_AVAILABLE.items() if v == sf]
    )

    print(f"\n  [{variant_label}] {model_label}  ({len(units)} units)")
    print(f"    (a) Neighbor hits     : {nbr_hits or 'none'}")
    print(f"    (b) Non-GT SF rows    : {[(t, sf) for t, sf, _ in non_gt_sf[:5]] or 'none'}")
    print(f"    (c) Available correct : {correct_c}/{len(GT_AVAILABLE)}  found={sorted(avail_found)}")
    for gt_un, gt_sf in sorted(GT_AVAILABLE.items()):
        if gt_sf in avail_found:
            ok = avail_found[gt_sf] == str(gt_un)
            tag = "✓" if ok else f"✗ model unit#{avail_found[gt_sf]}"
            print(f"         unit#{gt_un}={gt_sf}: {tag}")
        else:
            print(f"         unit#{gt_un}={gt_sf}: MISSING")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    variants = _build_variants()
    _assert_integrity(variants)

    client = get_openrouter_client()
    gemini = next(m for m in VISION_MODELS if m.name == "Gemini 3.7 Flash")
    qwen   = next(m for m in VISION_MODELS if m.name == "Qwen3-VL 32B")
    pdf    = PDF_PATH

    print("=" * 70)
    print("ISOLATION TEST — Gemini 3.7 Flash (native PDF)")
    print("=" * 70)
    for variant_name, prompt in variants.items():
        print(f"\n  Running '{variant_name}'...", end="", flush=True)
        result = _run_native_pdf(client, gemini, pdf, prompt)
        _score(result, "Gemini 3.7 Flash", variant_name)

    print("\n\n" + "=" * 70)
    print("ISOLATION TEST — Qwen3-VL 32B (rasterized)")
    print("=" * 70)
    for variant_name, prompt in variants.items():
        print(f"\n  Running '{variant_name}'...", end="", flush=True)
        result = _run_rasterized(client, qwen, pdf, prompt)
        _score(result, "Qwen3-VL 32B", variant_name)

    print("\n\nDone.")


if __name__ == "__main__":
    main()
