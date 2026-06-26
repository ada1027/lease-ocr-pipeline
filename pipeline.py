"""Lease OCR pipeline — ingest, classify, extract text, write to CSV."""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.extraction.ocr import extract_scanned_text
from src.extraction.searchable import extract_candidate_text
from src.ingestion.loader import load_pdf
from src.utils.logging import get_logger

load_dotenv()
logger = get_logger(__name__)

CSV_COLUMNS = [
    "store_id", "filename", "total_pages", "searchable_pages", "scanned_pages",
    "candidate_chars", "candidate_text", "extracted_at",
]


def run(pdf_path: str, store_id: str) -> None:
    path = Path(pdf_path)

    # 1. Ingest + validate + classify pages
    doc = load_pdf(pdf_path)

    # 2. Extract text from searchable pages
    candidate_text = extract_candidate_text(doc.pages)

    # 3. OCR scanned pages (stub — appends when opendataloader-pdf is wired in)
    try:
        ocr_text = extract_scanned_text(doc.pages, pdf_path)
        if ocr_text:
            candidate_text = candidate_text + "\n\n" + ocr_text
    except NotImplementedError:
        logger.warning("OCR step skipped — opendataloader-pdf not yet integrated.")

    searchable = sum(1 for p in doc.pages if p.mode == "searchable")
    scanned = sum(1 for p in doc.pages if p.mode == "scanned")

    # 4. Write to CSV
    out_dir = Path(os.getenv("OUTPUT_DIR", "data/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "enrichment_table.csv"

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
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

    logger.info("Written to %s", csv_path)
    print(f"Done — {path.name}: {doc.page_count} pages ({searchable} searchable, {scanned} scanned), {len(candidate_text)} chars → {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lease OCR extraction pipeline")
    parser.add_argument("pdf", help="Path to the lease PDF")
    parser.add_argument("store_id", help="Sitewise store identifier")
    args = parser.parse_args()
    run(args.pdf, args.store_id)


if __name__ == "__main__":
    main()
