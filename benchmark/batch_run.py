"""
Batch extraction runner — runs Gemini 3.7 Flash across every PDF in data/,
applies the post-processing confidence filter, and writes a summary report.

Usage:
    .venv/bin/python benchmark/batch_run.py [--limit N] [--resume]

Options:
    --limit N    Process only the first N documents (for a quick smoke-test)
    --resume     Skip documents already present in the latest run CSV

Quality signals (no manual GT needed):
    citation_ok_rate     — fraction of units where Check 1/2/3 all passed
    unverified_rate      — fraction flagged as fabricated by Check 1
    null_sf_rate         — fraction with square_footage=null (tenant known, SF unknown)
    unit_count           — raw units before filtering
    kept_units           — units that pass the confidence filter
    doc_type             — classification assigned by the model
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.vision_run import (
    VISION_MODELS,
    call_openrouter_native_pdf,
    call_openrouter_vision,
    get_openrouter_client,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_REPO_ROOT  = Path(__file__).resolve().parent.parent
_DATA_DIR   = _REPO_ROOT / "data"
_OUTPUT_DIR = _REPO_ROOT / "benchmark" / "results"

# Model to use for batch run — Gemini 3.7 Flash (native PDF, best accuracy on this doc type)
_BATCH_MODEL_NAME = "Gemini 3.7 Flash"

# Post-processing filter: only keep units where citation_status is one of these
_KEEP_STATUSES = {"ok", "unchecked", "ambiguous_assignment"}
# "unverified_citation" and "suspect_assignment" are dropped from kept_units count
# (still appear in raw output so nothing is lost)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_pdfs() -> list[Path]:
    return sorted(_DATA_DIR.rglob("*.pdf"))


def _citation_stats(units: list[dict]) -> dict:
    if not units:
        return {
            "unit_count": 0, "kept_units": 0,
            "citation_ok_rate": None, "unverified_rate": None, "null_sf_rate": None,
        }
    total = len(units)
    ok      = sum(1 for u in units if u.get("citation_status") == "ok")
    unver   = sum(1 for u in units if u.get("citation_status") == "unverified_citation")
    null_sf = sum(1 for u in units if u.get("square_footage") is None)
    kept    = sum(1 for u in units if u.get("citation_status") in _KEEP_STATUSES)
    return {
        "unit_count":        total,
        "kept_units":        kept,
        "citation_ok_rate":  round(ok   / total, 3),
        "unverified_rate":   round(unver / total, 3),
        "null_sf_rate":      round(null_sf / total, 3),
    }


def _already_done(run_csv: Path) -> set[str]:
    """Return set of pdf filenames already in the run CSV (for --resume)."""
    done: set[str] = set()
    if not run_csv.exists():
        return done
    with run_csv.open() as f:
        for row in csv.DictReader(f):
            if row.get("pdf"):
                done.add(row["pdf"])
    return done


def _write_summary_csv(summary_rows: list[dict], path: Path) -> None:
    if not summary_rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def _write_detail_csv(detail_rows: list[dict], path: Path) -> None:
    if not detail_rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)


def _write_text_report(summary_rows: list[dict], path: Path, total_cost: float, elapsed: float) -> None:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("BATCH EXTRACTION REPORT")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"Model: {_BATCH_MODEL_NAME} (native PDF)")
    lines.append(f"Documents processed: {len(summary_rows)}")
    lines.append(f"Total cost: ${total_cost:.4f}")
    lines.append(f"Total time: {elapsed/60:.1f} min")
    lines.append("=" * 80)
    lines.append("")

    # Aggregate stats
    rows_with_units = [r for r in summary_rows if r["unit_count"] > 0]
    all_units   = sum(r["unit_count"]  for r in summary_rows)
    all_kept    = sum(r["kept_units"]  for r in summary_rows)
    valid_count = sum(1 for r in summary_rows if r["json_valid"])

    lines.append("AGGREGATE SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Valid JSON responses : {valid_count}/{len(summary_rows)} ({100*valid_count//max(len(summary_rows),1)}%)")
    lines.append(f"  Total units extracted: {all_units}")
    lines.append(f"  Kept after filter    : {all_kept} ({100*all_kept//max(all_units,1)}%)")
    lines.append(f"  Avg units/doc        : {all_units/max(len(summary_rows),1):.1f}")
    lines.append(f"  Avg kept/doc         : {all_kept/max(len(summary_rows),1):.1f}")
    lines.append("")

    # Doc-type breakdown
    by_type: dict[str, int] = defaultdict(int)
    for r in summary_rows:
        by_type[r.get("doc_type") or "unknown"] += 1
    lines.append("DOC TYPE BREAKDOWN")
    lines.append("-" * 40)
    for dt, count in sorted(by_type.items(), key=lambda x: -x[1]):
        lines.append(f"  {dt:<25} {count:>4} docs")
    lines.append("")

    # Quality tiers
    high_conf  = [r for r in rows_with_units if (r.get("citation_ok_rate") or 0) >= 0.8]
    mixed_conf = [r for r in rows_with_units if 0.3 <= (r.get("citation_ok_rate") or 0) < 0.8]
    low_conf   = [r for r in rows_with_units if (r.get("citation_ok_rate") or 0) < 0.3]
    empty_docs = [r for r in summary_rows if r["unit_count"] == 0]

    lines.append("QUALITY TIERS  (citation_ok_rate threshold)")
    lines.append("-" * 40)
    lines.append(f"  High confidence  (≥80% ok) : {len(high_conf):>4} docs")
    lines.append(f"  Mixed confidence (30-79%)  : {len(mixed_conf):>4} docs")
    lines.append(f"  Low confidence   (<30%)    : {len(low_conf):>4} docs")
    lines.append(f"  No units extracted         : {len(empty_docs):>4} docs")
    lines.append("")

    # High-fabrication docs (>50% unverified)
    fab_docs = [r for r in rows_with_units if (r.get("unverified_rate") or 0) > 0.5]
    if fab_docs:
        lines.append(f"HIGH-FABRICATION DOCS  (>50% unverified_citation)  — {len(fab_docs)} docs")
        lines.append("-" * 40)
        for r in sorted(fab_docs, key=lambda x: -(x.get("unverified_rate") or 0))[:20]:
            lines.append(
                f"  {r['site_id']:<10}  unverified={r.get('unverified_rate',0):.0%}"
                f"  units={r['unit_count']}  doc_type={r.get('doc_type','?')}"
            )
        lines.append("")

    # Per-document table (all docs)
    lines.append("PER-DOCUMENT RESULTS")
    lines.append("-" * 80)
    hdr = f"  {'site_id':<10} {'doc_type':<20} {'units':>5} {'kept':>5} {'ok%':>5} {'unver%':>6} {'null_sf%':>8} {'cost':>8}"
    lines.append(hdr)
    lines.append("  " + "-" * 76)
    for r in sorted(summary_rows, key=lambda x: x.get("site_id", "")):
        ok_r   = r.get("citation_ok_rate")
        unv_r  = r.get("unverified_rate")
        nul_r  = r.get("null_sf_rate")
        lines.append(
            f"  {r.get('site_id','?'):<10} {(r.get('doc_type') or '?'):<20}"
            f" {r['unit_count']:>5} {r['kept_units']:>5}"
            f" {f'{ok_r:.0%}' if ok_r is not None else '  N/A':>5}"
            f" {f'{unv_r:.0%}' if unv_r is not None else '   N/A':>6}"
            f" {f'{nul_r:.0%}' if nul_r is not None else '     N/A':>8}"
            f" ${r.get('cost_usd',0):.4f}"
        )
    lines.append("")
    lines.append("=" * 80)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch extraction across all PDFs in data/")
    parser.add_argument("--limit",  type=int, default=None, help="Process only first N docs")
    parser.add_argument("--resume", action="store_true",    help="Skip already-processed docs")
    args = parser.parse_args()

    model = next((m for m in VISION_MODELS if m.name == _BATCH_MODEL_NAME), None)
    if model is None:
        print(f"ERROR: model '{_BATCH_MODEL_NAME}' not found in VISION_MODELS", file=sys.stderr)
        sys.exit(1)

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    _OUTPUT_DIR.mkdir(exist_ok=True)
    run_id  = datetime.now(timezone.utc).strftime("batch_%Y%m%d_%H%M%S")
    run_csv = _OUTPUT_DIR / f"{run_id}_summary.csv"
    det_csv = _OUTPUT_DIR / f"{run_id}_units.csv"
    rpt_txt = _OUTPUT_DIR / f"{run_id}_report.txt"

    all_pdfs = _all_pdfs()
    if args.limit:
        all_pdfs = all_pdfs[: args.limit]

    done_pdfs: set[str] = set()
    if args.resume:
        done_pdfs = _already_done(run_csv)
        skipped = sum(1 for p in all_pdfs if p.name in done_pdfs)
        print(f"--resume: skipping {skipped} already-processed docs")

    client = get_openrouter_client()

    summary_rows: list[dict] = []
    detail_rows:  list[dict] = []
    total_cost  = 0.0
    t_start     = time.perf_counter()

    todo = [p for p in all_pdfs if p.name not in done_pdfs]
    print(f"Batch run {run_id}")
    print(f"Model   : {model.name} (native_pdf={model.native_pdf})")
    print(f"Docs    : {len(todo)} to process  ({len(all_pdfs)} total, {len(done_pdfs)} skipped)")
    print(f"Output  : {run_csv.name}")
    print("-" * 60)

    for i, pdf_path in enumerate(todo, 1):
        # Extract site_id from directory name (data/SiteFiles.../SITEID/filename.pdf)
        parts = pdf_path.parts
        try:
            sf_idx = next(j for j, p in enumerate(parts) if p == "SiteFiles")
            site_id = parts[sf_idx + 1]
        except (StopIteration, IndexError):
            site_id = pdf_path.stem[:12]

        kb = pdf_path.stat().st_size // 1024
        print(f"  [{i:>3}/{len(todo)}] {site_id:<10} {pdf_path.name[:40]:<40} ({kb} KB) ... ",
              end="", flush=True)

        try:
            if model.native_pdf:
                result = call_openrouter_native_pdf(client, model, str(pdf_path), 50)
            else:
                result = call_openrouter_vision(client, model, str(pdf_path), 50)
        except Exception as exc:
            print(f"ERROR: {exc}")
            result = {
                "model_name": model.name, "json_valid": False,
                "schema_error": str(exc)[:120], "units": [],
                "doc_type": "error", "cost_usd": 0.0, "latency_sec": 0.0,
            }

        stats = _citation_stats(result.get("units", []))
        total_cost += result.get("cost_usd", 0.0)

        ok_str = f"{stats['citation_ok_rate']:.0%}" if stats['citation_ok_rate'] is not None else 'N/A'
        print(
            f"✓ {stats['unit_count']} units  kept={stats['kept_units']}"
            f"  ok={ok_str}"
            f"  ${result.get('cost_usd',0):.4f}  {result.get('latency_sec',0):.1f}s"
        )

        summary_row = {
            "site_id":           site_id,
            "pdf":               pdf_path.name,
            "pdf_path":          str(pdf_path.relative_to(_REPO_ROOT)),
            "model":             model.name,
            "json_valid":        result.get("json_valid", False),
            "schema_error":      result.get("schema_error", ""),
            "doc_type":          result.get("doc_type", "unknown"),
            "input_tokens":      result.get("input_tokens", 0),
            "output_tokens":     result.get("output_tokens", 0),
            "cost_usd":          result.get("cost_usd", 0.0),
            "latency_sec":       result.get("latency_sec", 0.0),
            **stats,
        }
        summary_rows.append(summary_row)

        for j, unit in enumerate(result.get("units", [])):
            kept = unit.get("citation_status") in _KEEP_STATUSES
            detail_rows.append({
                "site_id":         site_id,
                "pdf":             pdf_path.name,
                "unit_index":      j,
                "kept":            kept,
                "tenant_name":     unit.get("tenant_name"),
                "square_footage":  unit.get("square_footage"),
                "unit_number":     unit.get("unit_number"),
                "unit_type":       unit.get("unit_type"),
                "confidence":      unit.get("confidence"),
                "doc_type":        result.get("doc_type", "unknown"),
                "citation_status": unit.get("citation_status"),
                "citation_note":   unit.get("citation_note", ""),
                "evidence_quote":  (unit.get("evidence_quote") or "")[:200],
                "source_location": unit.get("source_location"),
            })

        # Flush both CSVs after every doc (safe to interrupt)
        _write_summary_csv(summary_rows, run_csv)
        _write_detail_csv(detail_rows, det_csv)

    elapsed = time.perf_counter() - t_start
    _write_text_report(summary_rows, rpt_txt, total_cost, elapsed)

    print(f"\n{'='*60}")
    print(f"Done. {len(summary_rows)} docs in {elapsed/60:.1f} min.  Total cost: ${total_cost:.4f}")
    print(f"Summary CSV : {run_csv}")
    print(f"Units CSV   : {det_csv}")
    print(f"Text report : {rpt_txt}")


if __name__ == "__main__":
    main()
