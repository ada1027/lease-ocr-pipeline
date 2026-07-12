"""Lease OCR extraction pipeline — single file or batch folder."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.extraction.ocr import extract_scanned_text
from src.extraction.searchable import extract_candidate_text
from src.ingestion.loader import load_pdf
from src.utils.logging import get_logger

load_dotenv()
logger = get_logger(__name__)

HAS_API_KEY = bool(os.getenv("OPENROUTER_API_KEY"))


def run_single(pdf_path: str, store_id: str) -> dict:
    """Run the full pipeline on one PDF. Returns a result dict for batch reporting."""
    path = Path(pdf_path)
    result = {"file": path.name, "store_id": store_id, "status": "ok", "error": None}

    try:
        # 1. Ingest + validate + classify pages
        doc = load_pdf(pdf_path)

        # 2. Extract text from searchable pages
        candidate_text = extract_candidate_text(doc.pages)

        # 3. OCR scanned pages and append
        ocr_text = extract_scanned_text(doc.pages, pdf_path)
        if ocr_text:
            candidate_text = candidate_text + "\n\n" + ocr_text

        searchable = sum(1 for p in doc.pages if p.mode == "searchable")
        scanned = sum(1 for p in doc.pages if p.mode == "scanned")

        if HAS_API_KEY:
            # 4. AI extraction
            from src.ai.extractor import extract_square_footage
            from src.ai.vision import extract_via_vision, find_floorplan_pages
            from src.persistence.enrichment import write_result

            ai_result = extract_square_footage(candidate_text)
            source = "text_extraction"

            # 5. Vision fallback if confidence is low
            if ai_result.confidence == "low":
                logger.info("%s: low confidence — trying Vision fallback.", path.name)
                floorplan_pages = find_floorplan_pages(doc.pages)
                if floorplan_pages:
                    ai_result = extract_via_vision(pdf_path, floorplan_pages)
                    source = "vision_fallback"
                else:
                    logger.warning("%s: no floorplan pages found for Vision fallback.", path.name)

            # 6. Write structured result to enrichment CSV
            written = write_result(store_id=store_id, result=ai_result, source=source)
            if written:
                result["square_footage"] = ai_result.square_footage
                result["confidence"] = ai_result.confidence
                result["source"] = source
                print(
                    f"✓ {path.name} → {ai_result.square_footage} {ai_result.unit} "
                    f"(confidence: {ai_result.confidence}, source: {source})"
                )
            else:
                result["status"] = "skipped"
                result["error"] = f"confidence too low ({ai_result.confidence}) or no square footage found"
                print(f"⚠ {path.name} — skipped: {result['error']}")

        else:
            # No API key — write raw extracted text to CSV
            import csv
            from datetime import datetime, timezone

            out_dir = Path(os.getenv("OUTPUT_DIR", "data/output"))
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / "enrichment_table.csv"

            columns = [
                "store_id", "filename", "total_pages", "searchable_pages",
                "scanned_pages", "candidate_chars", "candidate_text", "extracted_at",
            ]
            write_header = not csv_path.exists()
            with csv_path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=columns)
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "store_id": store_id,
                    "filename": path.name,
                    "total_pages": doc.page_count,
                    "searchable_pages": searchable,
                    "scanned_pages": scanned,
                    "candidate_chars": len(candidate_text),
                    "candidate_text": candidate_text,
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                })

            print(
                f"✓ {path.name}: {doc.page_count} pages "
                f"({searchable} searchable, {scanned} scanned), "
                f"{len(candidate_text)} chars → {csv_path}"
            )
            print("  (Add OPENROUTER_API_KEY to .env to enable AI extraction)")

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error("Failed on %s: %s", path.name, e)
        print(f"✗ {path.name} — ERROR: {e}")

    return result


def run_batch(folder: str) -> None:
    """Run the pipeline on every PDF in a folder and print a summary report."""
    pdfs = sorted(Path(folder).glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {folder}")
        sys.exit(1)

    print(f"\nBatch: {len(pdfs)} PDF(s) found in {folder}\n{'-'*50}")

    results = []
    for pdf in pdfs:
        store_id = pdf.stem.upper().replace(" ", "-")
        results.append(run_single(str(pdf), store_id))

    # Summary report
    ok      = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed  = [r for r in results if r["status"] == "failed"]

    print(f"\n{'='*50}")
    print(f"BATCH COMPLETE")
    print(f"  Total:   {len(results)}")
    print(f"  Success: {len(ok)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"  Failed:  {len(failed)}")

    if skipped:
        print("\nSkipped (low confidence or no square footage found):")
        for r in skipped:
            print(f"  - {r['file']}: {r['error']}")

    if failed:
        print("\nFailed (pipeline error):")
        for r in failed:
            print(f"  - {r['file']}: {r['error']}")

    print(f"{'='*50}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lease OCR extraction pipeline")
    parser.add_argument("path", help="Path to a single PDF or a folder (use with --batch)")
    parser.add_argument("store_id", nargs="?", help="Sitewise store ID (single file mode only)")
    parser.add_argument("--batch", action="store_true", help="Process every PDF in a folder")
    args = parser.parse_args()

    if args.batch:
        run_batch(args.path)
    else:
        if not args.store_id:
            parser.error("store_id is required in single file mode")
        run_single(args.path, args.store_id)


if __name__ == "__main__":
    main()
