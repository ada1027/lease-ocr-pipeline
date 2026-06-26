"""Unit tests for PDF ingestion and mode detection."""

import pytest
from unittest.mock import MagicMock, patch

from src.ingestion.loader import (
    SEARCHABLE_CHAR_THRESHOLD,
    _validate_page_count,
    detect_page_mode,
)


def test_validate_page_count_valid():
    _validate_page_count(10, "test.pdf")  # should not raise


def test_validate_page_count_too_few():
    with pytest.raises(ValueError, match="minimum"):
        _validate_page_count(1, "test.pdf")


def test_validate_page_count_too_many():
    with pytest.raises(ValueError, match="maximum"):
        _validate_page_count(201, "test.pdf")


def _make_mock_page(text: str) -> MagicMock:
    """Return a fitz.Page mock that returns the given text from get_text('blocks')."""
    page = MagicMock()
    page.number = 0
    # blocks format: (x0, y0, x1, y1, text, block_no, block_type)
    page.get_text.return_value = [(0, 0, 100, 20, text, 0, 0)]
    return page


def test_detect_page_mode_searchable():
    rich_text = "a" * SEARCHABLE_CHAR_THRESHOLD
    page = _make_mock_page(rich_text)
    mode, char_count, _ = detect_page_mode(page)
    assert mode == "searchable"
    assert char_count >= SEARCHABLE_CHAR_THRESHOLD


def test_detect_page_mode_scanned():
    sparse_text = "a" * (SEARCHABLE_CHAR_THRESHOLD - 1)
    page = _make_mock_page(sparse_text)
    mode, char_count, _ = detect_page_mode(page)
    assert mode == "scanned"
