"""OCR extraction for scanned pages using PyMuPDF rasterization + Tesseract."""

from __future__ import annotations

import io
from typing import List

import fitz
import pytesseract
from PIL import Image

from src.ingestion.loader import PageInfo
from src.utils.logging import get_logger

logger = get_logger(__name__)

OCR_DPI = 200


def extract_scanned_text(pages: List[PageInfo], pdf_path: str) -> str:
    """Run Tesseract OCR on all scanned pages and return combined text."""
    scanned = [p for p in pages if p.mode == "scanned"]
    if not scanned:
        logger.info("No scanned pages — OCR step skipped.")
        return ""

    logger.info("Running OCR on %d scanned page(s) in %s.", len(scanned), pdf_path)

    doc = fitz.open(pdf_path)
    results: list[str] = []

    for page_info in scanned:
        text = _ocr_page(doc, page_info.page_number)
        if text.strip():
            results.append(f"--- Page {page_info.page_number} (OCR) ---\n{text.strip()}")
            logger.debug("Page %d OCR: %d chars extracted.", page_info.page_number, len(text))
        else:
            logger.warning("Page %d OCR returned no text.", page_info.page_number)

    doc.close()

    combined = "\n\n".join(results)
    logger.info("OCR complete — %d chars extracted across %d page(s).", len(combined), len(scanned))
    return combined


def rasterize_page(pdf_path: str, page_number: int, dpi: int = OCR_DPI) -> bytes:
    """Return PNG bytes for a single page at the given DPI. Used by the Vision fallback."""
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    doc.close()
    return pix.tobytes("png")


def _ocr_page(doc: fitz.Document, page_number: int) -> str:
    """Rasterize a single page and run Tesseract on it."""
    page = doc[page_number - 1]
    mat = fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)
