"""PDF ingestion: load, validate page count, and route pages to the correct extractor."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from src.utils.logging import get_logger

logger = get_logger(__name__)

MIN_PAGES = 2
MAX_PAGES = 200

# Characters of extracted text below this threshold → treat page as scanned.
# Tune empirically; typical lease pages with real text yield 200–800+ chars.
SEARCHABLE_CHAR_THRESHOLD = 100


@dataclass
class PageInfo:
    page_number: int          # 1-based
    mode: str                 # "searchable" | "scanned"
    char_count: int
    text: str = field(repr=False, default="")


@dataclass
class IngestedDocument:
    path: Path
    page_count: int
    pages: List[PageInfo] = field(repr=False, default_factory=list)


def load_pdf(pdf_path: str | Path) -> IngestedDocument:
    """Load a lease PDF, validate page count, classify each page, and return a document object."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    logger.info("Opening PDF: %s", path)
    doc = fitz.open(str(path))
    page_count = len(doc)
    logger.info("Page count: %d", page_count)

    _validate_page_count(page_count, path)

    pages = [_classify_page(doc[i], i + 1) for i in range(page_count)]
    doc.close()

    searchable = sum(1 for p in pages if p.mode == "searchable")
    scanned = page_count - searchable
    logger.info(
        "Classification complete — searchable: %d, scanned: %d", searchable, scanned
    )

    return IngestedDocument(path=path, page_count=page_count, pages=pages)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_page_count(count: int, path: "str | Path") -> None:
    name = Path(path).name
    if count < MIN_PAGES:
        raise ValueError(f"PDF '{name}' has {count} page(s); minimum is {MIN_PAGES}.")
    if count > MAX_PAGES:
        raise ValueError(f"PDF '{name}' has {count} pages; maximum is {MAX_PAGES}.")
    logger.debug("Page count %d is within valid range [%d, %d].", count, MIN_PAGES, MAX_PAGES)


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

def detect_page_mode(page: fitz.Page) -> tuple[str, int, str]:
    """Classify a single PyMuPDF page as 'searchable' or 'scanned'.

    Returns (mode, char_count, extracted_text).

    Strategy:
      - Extract text in 'blocks' mode (preserves spatial layout).
      - Count non-whitespace characters.
      - Pages below SEARCHABLE_CHAR_THRESHOLD are treated as scanned images
        that require OCR via the opendataloader-pdf workflow.
    """
    blocks = page.get_text("blocks")  # list of (x0,y0,x1,y1,text,block_no,block_type)
    text = "\n".join(b[4] for b in blocks if b[6] == 0)  # block_type 0 = text
    char_count = len(text.replace(" ", "").replace("\n", ""))

    mode = "searchable" if char_count >= SEARCHABLE_CHAR_THRESHOLD else "scanned"
    logger.debug(
        "Page %s: mode=%s, non-ws chars=%d", page.number + 1, mode, char_count
    )
    return mode, char_count, text


def _classify_page(page: fitz.Page, page_number: int) -> PageInfo:
    mode, char_count, text = detect_page_mode(page)
    return PageInfo(
        page_number=page_number,
        mode=mode,
        char_count=char_count,
        text=text,
    )
