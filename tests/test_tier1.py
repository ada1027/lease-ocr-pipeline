"""Tier 1 pipeline tests — extractor schema, enrichment CSV, and end-to-end on real PDFs."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ai.extractor import ExtractionResult, _fallback_result, _strip_code_block
from src.persistence.enrichment import CSV_COLUMNS, write_result


# ---------------------------------------------------------------------------
# ExtractionResult schema
# ---------------------------------------------------------------------------

def test_extraction_result_defaults():
    r = ExtractionResult(square_footage=1850, unit="sq ft", confidence=0.92, evidence="test")
    assert r.doc_type == "unknown"
    assert r.source_tag == "lease_ocr"
    assert r.tenant_name is None
    assert r.suite_number is None
    assert isinstance(r.confidence, float)
    assert r.extracted_at  # non-empty ISO timestamp


def test_extraction_result_confidence_is_float():
    r = ExtractionResult(square_footage=500, unit="sq ft", confidence=0.75, evidence="x")
    assert isinstance(r.confidence, float)
    assert 0.0 <= r.confidence <= 1.0


def test_fallback_result_never_null():
    r = _fallback_result("no text found")
    assert r.square_footage == 0   # 0 not None
    assert r.confidence == 0.1
    assert r.source_tag == "lease_ocr"


def test_source_tags_are_valid():
    valid = {"lease_ocr", "leasing_pdf", "mall_directory", "vision_estimate",
             "overture_footprint", "osm_footprint", "model_predicted", "brand_average_fallback"}
    r = ExtractionResult(square_footage=1000, unit="sq ft", confidence=0.9,
                         evidence="x", source_tag="leasing_pdf")
    assert r.source_tag in valid


# ---------------------------------------------------------------------------
# Output schema matches proposal
# ---------------------------------------------------------------------------

PROPOSAL_COLUMNS = {"store_id", "square_footage", "source", "confidence", "evidence", "extracted_at"}

def test_csv_columns_include_proposal_schema():
    """Every field in the proposal output schema must appear in the CSV."""
    assert PROPOSAL_COLUMNS.issubset(set(CSV_COLUMNS)), (
        f"Missing columns: {PROPOSAL_COLUMNS - set(CSV_COLUMNS)}"
    )


def test_csv_no_legacy_column():
    assert "evidence_snippet" not in CSV_COLUMNS


# ---------------------------------------------------------------------------
# Enrichment writer — always writes (100% coverage)
# ---------------------------------------------------------------------------

def test_write_result_always_writes(tmp_path):
    r = ExtractionResult(square_footage=0, unit=None, confidence=0.1,
                         evidence="nothing found", source_tag="lease_ocr")
    written = write_result(store_id="STORE-001", result=r, output_dir=str(tmp_path))
    assert written is True
    rows = list(csv.DictReader((tmp_path / "enrichment_table.csv").open()))
    assert len(rows) == 1
    assert rows[0]["square_footage"] == "0"
    assert rows[0]["confidence"] == "0.1"


def test_write_result_writes_all_proposal_fields(tmp_path):
    r = ExtractionResult(
        square_footage=1850, unit="sq ft", confidence=0.92,
        evidence="file:///path/to.pdf — '1,850 sq ft demised premises'",
        doc_type="executed_lease", tenant_name="Chipotle Mexican Grill",
        suite_number="Suite 140", source_tag="lease_ocr",
    )
    write_result(store_id="STORE-XYZ", result=r, output_dir=str(tmp_path))
    row = list(csv.DictReader((tmp_path / "enrichment_table.csv").open()))[0]

    assert row["store_id"] == "STORE-XYZ"
    assert row["square_footage"] == "1850"
    assert row["confidence"] == "0.92"
    assert row["source"] == "lease_ocr"
    assert "file:///path/to.pdf" in row["evidence"]
    assert row["doc_type"] == "executed_lease"
    assert row["tenant_name"] == "Chipotle Mexican Grill"
    assert row["suite_number"] == "Suite 140"
    assert row["extracted_at"]


def test_write_result_100pct_coverage_low_confidence(tmp_path):
    """Low-confidence results must still be written — never skipped."""
    r = ExtractionResult(square_footage=0, unit=None, confidence=0.1, evidence="no SF found")
    written = write_result(store_id="S", result=r, output_dir=str(tmp_path))
    assert written is True


# ---------------------------------------------------------------------------
# Extractor JSON parsing — mocked API
# ---------------------------------------------------------------------------

MOCK_RESPONSE_FLYER = """{
  "doc_type": "leasing_flyer",
  "square_footage": 46444,
  "unit": "sq ft",
  "confidence": 0.95,
  "tenant_name": "Bainbridge Marketplace",
  "suite_number": null,
  "source_tag": "leasing_pdf",
  "evidence": "46,444 SF Food Lion-anchored shopping center"
}"""

MOCK_RESPONSE_LEASE = """{
  "doc_type": "executed_lease",
  "square_footage": 1850,
  "unit": "sq ft",
  "confidence": 0.93,
  "tenant_name": "Chipotle Mexican Grill",
  "suite_number": "Suite 140",
  "source_tag": "lease_ocr",
  "evidence": "Demised Premises: 1,850 square feet known as Suite 140"
}"""


def _mock_openai_response(json_text: str):
    choice = MagicMock()
    choice.message.content = json_text
    response = MagicMock()
    response.choices = [choice]
    return response


@patch("src.ai.extractor.get_client")
def test_extract_leasing_flyer(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(MOCK_RESPONSE_FLYER)
    mock_get_client.return_value = mock_client

    from src.ai.extractor import extract_square_footage
    result = extract_square_footage("some flyer text")

    assert result.doc_type == "leasing_flyer"
    assert result.square_footage == 46444
    assert result.confidence == 0.95
    assert result.source_tag == "leasing_pdf"
    assert result.tenant_name == "Bainbridge Marketplace"
    assert result.suite_number is None


@patch("src.ai.extractor.get_client")
def test_extract_executed_lease(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(MOCK_RESPONSE_LEASE)
    mock_get_client.return_value = mock_client

    from src.ai.extractor import extract_square_footage
    result = extract_square_footage("some lease text")

    assert result.doc_type == "executed_lease"
    assert result.square_footage == 1850
    assert result.confidence == 0.93
    assert result.source_tag == "lease_ocr"
    assert result.tenant_name == "Chipotle Mexican Grill"
    assert result.suite_number == "Suite 140"


@patch("src.ai.extractor.get_client")
def test_extract_bad_json_returns_fallback(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response("not json at all")
    mock_get_client.return_value = mock_client

    from src.ai.extractor import extract_square_footage
    result = extract_square_footage("some text")

    assert result.square_footage == 0   # fallback, never None
    assert result.confidence == 0.1


@patch("src.ai.extractor.get_client")
def test_extract_empty_text_skips_api(mock_get_client):
    from src.ai.extractor import extract_square_footage
    result = extract_square_footage("   ")
    mock_get_client.assert_not_called()
    assert result.square_footage == 0
    assert result.confidence == 0.1


# ---------------------------------------------------------------------------
# Vision extractor — source tag override
# ---------------------------------------------------------------------------

MOCK_VISION_RESPONSE = """{
  "doc_type": "leasing_flyer",
  "square_footage": 36500,
  "unit": "sq ft",
  "confidence": 0.80,
  "tenant_name": "Choctaw Village",
  "suite_number": null,
  "evidence": "Site plan shows 36,500 SF total"
}"""


@patch("src.ai.vision.get_client")
@patch("src.ai.vision.rasterize_page")
def test_vision_source_tag_is_vision_estimate(mock_rasterize, mock_get_client):
    mock_rasterize.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(MOCK_VISION_RESPONSE)
    mock_get_client.return_value = mock_client

    from src.ai.vision import extract_via_vision
    result = extract_via_vision("fake.pdf", [1])

    assert result.source_tag == "vision_estimate"
    assert result.square_footage == 36500


# ---------------------------------------------------------------------------
# End-to-end on real PDFs (skipped if no API key)
# ---------------------------------------------------------------------------

REAL_PDFS = [
    ("data/village_of_blaine.pdf",  "STORE-VB",  "leasing_flyer",  "leasing_pdf",   200000, 250000),
    ("data/Market at Darrington.pdf", "STORE-MD", None,             None,            0,      500000),
]


@pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="No API key")
@pytest.mark.parametrize("pdf,store_id,expected_doc_type,expected_source,sf_min,sf_max", REAL_PDFS)
def test_end_to_end_real_pdf(tmp_path, pdf, store_id, expected_doc_type, expected_source, sf_min, sf_max):
    from src.ingestion.loader import load_pdf
    from src.extraction.ocr import extract_scanned_text
    from src.extraction.searchable import extract_candidate_text
    from src.ai.extractor import extract_square_footage

    if not Path(pdf).exists():
        pytest.skip(f"PDF not found: {pdf}")

    doc = load_pdf(pdf)
    text = extract_candidate_text(doc.pages)
    ocr = extract_scanned_text(doc.pages, pdf)
    if ocr:
        text = text + "\n\n" + ocr

    result = extract_square_footage(text)

    # Schema checks — always
    assert isinstance(result.square_footage, int), "square_footage must be int"
    assert isinstance(result.confidence, float), "confidence must be float"
    assert 0.0 <= result.confidence <= 1.0, "confidence must be in [0, 1]"
    assert result.source_tag in {
        "lease_ocr", "leasing_pdf", "mall_directory"
    }, f"unexpected source_tag: {result.source_tag}"
    assert result.evidence, "evidence must not be empty"
    assert result.doc_type in {
        "executed_lease", "leasing_flyer", "tenant_roster", "unknown"
    }

    # Value checks
    assert sf_min <= result.square_footage <= sf_max, (
        f"SF {result.square_footage} outside expected [{sf_min}, {sf_max}]"
    )
    if expected_doc_type:
        assert result.doc_type == expected_doc_type
    if expected_source:
        assert result.source_tag == expected_source

    # Write check — must always succeed
    written = write_result(store_id=store_id, result=result, output_dir=str(tmp_path))
    assert written is True
    rows = list(csv.DictReader((tmp_path / "enrichment_table.csv").open()))
    assert rows[0]["store_id"] == store_id
