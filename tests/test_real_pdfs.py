"""Integration tests against real PDF files in data/.

Run with:
    pytest tests/test_real_pdfs.py -v

PDFs are discovered automatically — drop any .pdf into data/ and re-run.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from src.extraction.ocr import extract_scanned_text
from src.extraction.searchable import extract_candidate_text
from src.ingestion.loader import load_pdf

DATA_DIR = Path(__file__).parent.parent / "data"


def _pdf_files():
    return sorted(DATA_DIR.glob("*.pdf"))


def pytest_generate_tests(metafunc):
    if "pdf_path" in metafunc.fixturenames:
        pdfs = _pdf_files()
        metafunc.parametrize("pdf_path", pdfs, ids=[p.name for p in pdfs])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pdfs_exist():
    pdfs = _pdf_files()
    if not pdfs:
        pytest.skip(f"No PDFs found in {DATA_DIR} — drop lease PDFs there and re-run.")
    assert len(pdfs) > 0


def test_ingestion(pdf_path):
    """PDF loads without error and page count is within 2–200."""
    doc = load_pdf(pdf_path)
    assert 2 <= doc.page_count <= 200
    assert len(doc.pages) == doc.page_count


def test_page_classification(pdf_path):
    """Every page is classified as searchable or scanned."""
    doc = load_pdf(pdf_path)
    for page in doc.pages:
        assert page.mode in ("searchable", "scanned")


def test_candidate_text_extraction(pdf_path):
    """Text extraction runs without error and returns a string."""
    doc = load_pdf(pdf_path)
    text = extract_candidate_text(doc.pages)
    assert isinstance(text, str)
    searchable = sum(1 for p in doc.pages if p.mode == "searchable")
    scanned = sum(1 for p in doc.pages if p.mode == "scanned")
    print(f"\n  {pdf_path.name}: {doc.page_count} pages ({searchable} searchable, {scanned} scanned), {len(text)} candidate chars")


def test_ocr_scanned_pages(pdf_path):
    """Scanned pages produce non-empty OCR text."""
    doc = load_pdf(str(pdf_path))
    scanned = [p for p in doc.pages if p.mode == "scanned"]
    if not scanned:
        pytest.skip(f"{pdf_path.name}: no scanned pages.")
    text = extract_scanned_text(doc.pages, str(pdf_path))
    assert isinstance(text, str)
    assert len(text.strip()) > 0, f"{pdf_path.name}: OCR returned empty text on {len(scanned)} scanned page(s)"
    print(f"\n  {pdf_path.name}: OCR extracted {len(text)} chars from {len(scanned)} scanned page(s)")


def test_combined_text_non_empty(pdf_path):
    """At least one extraction method (searchable or OCR) produces text."""
    doc = load_pdf(str(pdf_path))
    searchable_text = extract_candidate_text(doc.pages)
    ocr_text = extract_scanned_text(doc.pages, str(pdf_path))
    combined = (searchable_text + ocr_text).strip()
    assert len(combined) > 0, f"{pdf_path.name}: both searchable extraction and OCR returned empty text"
