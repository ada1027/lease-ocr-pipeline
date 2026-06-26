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

CONFIDENCE_THRESHOLD = "medium"
CSV_COLUMNS = ["store_id", "square_footage", "unit", "confidence", "evidence_snippet", "source", "extracted_at"]


def write_result(
    store_id: str,
    result: ExtractionResult,
    source: str,
    output_dir: Optional[str] = None,
) -> bool:
    """Append a valid extraction result to enrichment_table.csv.

    Returns True if written, False if skipped (low confidence or missing square_footage).
    """
    if result.square_footage is None:
        logger.warning("store_id=%s: square_footage is None — skipping.", store_id)
        return False

    confidence_rank = {"high": 2, "medium": 1, "low": 0}
    if confidence_rank.get(result.confidence, 0) < confidence_rank[CONFIDENCE_THRESHOLD]:
        logger.warning(
            "store_id=%s: confidence '%s' below threshold '%s' — skipping.",
            store_id, result.confidence, CONFIDENCE_THRESHOLD,
        )
        return False

    record = {
        "store_id": store_id,
        "square_footage": result.square_footage,
        "unit": result.unit,
        "confidence": result.confidence,
        "evidence_snippet": result.evidence_snippet,
        "source": source,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
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
        "Wrote record for store_id=%s: %s %s (confidence=%s) → %s",
        store_id, result.square_footage, result.unit, result.confidence, csv_path,
    )
    return True
