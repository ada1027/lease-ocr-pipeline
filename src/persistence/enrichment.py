"""Write extraction results to the enrichment CSV table."""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.ai.extractor import ExtractionResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

CSV_COLUMNS = [
    "store_id", "square_footage", "unit", "confidence",
    "doc_type", "tenant_name", "suite_number",
    "source", "evidence", "extracted_at",
]


def write_result(
    store_id: str,
    result: ExtractionResult,
    source: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> bool:
    """Append extraction result to enrichment_table.csv. Always writes (100% coverage).

    Returns True. The source column prefers result.source_tag; caller can override via `source`.
    """
    record = {
        "store_id": store_id,
        "square_footage": result.square_footage,
        "unit": result.unit or "sq ft",
        "confidence": result.confidence,
        "doc_type": result.doc_type,
        "tenant_name": result.tenant_name or "",
        "suite_number": result.suite_number or "",
        "source": source or result.source_tag,
        "evidence": result.evidence,
        "extracted_at": result.extracted_at,
    }

    out_dir = Path(output_dir or os.getenv("OUTPUT_DIR", "data/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "enrichment_table.csv"

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)

    logger.info(
        "Wrote store_id=%s: %s sq ft (doc_type=%s, confidence=%s, source=%s)",
        store_id, result.square_footage, result.doc_type, result.confidence, record["source"],
    )
    return True
