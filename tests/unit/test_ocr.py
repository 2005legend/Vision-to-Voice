"""Unit tests for board_reader.ocr."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from board_reader.ocr import combine_ocr, extract_latex, extract_text


# ---------------------------------------------------------------------------
# combine_ocr tests
# ---------------------------------------------------------------------------

def test_combine_ocr_both_non_empty():
    """combine_ocr with both non-empty inputs contains both substrings."""
    result = combine_ocr("Hello world", r"E = mc^2")
    assert "Hello world" in result
    assert r"E = mc^2" in result
    assert "[TEXT]" in result
    assert "[LATEX]" in result


def test_combine_ocr_text_empty():
    """combine_ocr with empty text still includes [TEXT] and [LATEX] sections."""
    result = combine_ocr("", r"\frac{a}{b}")
    assert "[TEXT]" in result
    assert "[LATEX]" in result
    assert r"\frac{a}{b}" in result


def test_combine_ocr_latex_empty():
    """combine_ocr with empty latex still includes both sections."""
    result = combine_ocr("Some text", "")
    assert "[TEXT]" in result
    assert "[LATEX]" in result
    assert "Some text" in result


def test_combine_ocr_both_empty():
    """combine_ocr with both empty still returns a string with both section headers."""
    result = combine_ocr("", "")
    assert isinstance(result, str)
    assert "[TEXT]" in result
    assert "[LATEX]" in result


def test_combine_ocr_returns_str():
    """combine_ocr always returns a str."""
    assert isinstance(combine_ocr("a", "b"), str)


def test_combine_ocr_structure():
    """combine_ocr output has [TEXT] before [LATEX]."""
    result = combine_ocr("text_part", "latex_part")
    text_pos = result.index("[TEXT]")
    latex_pos = result.index("[LATEX]")
    assert text_pos < latex_pos


# ---------------------------------------------------------------------------
# extract_text error-handling tests
# ---------------------------------------------------------------------------

def test_extract_text_returns_empty_string_when_paddle_raises():
    """extract_text returns '' when PaddleOCR raises an exception."""
    import board_reader.ocr as ocr_module

    with patch.object(ocr_module, "_get_paddle_ocr", side_effect=Exception("PaddleOCR failed")):
        result = extract_text(np.zeros((10, 10, 3), dtype=np.uint8))

    assert result == ""


def test_extract_text_returns_empty_string_when_ocr_returns_none():
    """extract_text returns '' when PaddleOCR returns None."""
    import board_reader.ocr as ocr_module

    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = None

    with patch.object(ocr_module, "_get_paddle_ocr", return_value=mock_ocr):
        result = extract_text(np.zeros((10, 10, 3), dtype=np.uint8))

    assert result == ""


def test_extract_text_returns_joined_lines():
    """extract_text joins detected text lines with newlines."""
    import board_reader.ocr as ocr_module

    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = [
        [
            [None, ("Hello", 0.99)],
            [None, ("World", 0.95)],
        ]
    ]

    with patch.object(ocr_module, "_get_paddle_ocr", return_value=mock_ocr):
        result = extract_text(np.zeros((10, 10, 3), dtype=np.uint8))

    assert "Hello" in result
    assert "World" in result


# ---------------------------------------------------------------------------
# extract_latex error-handling tests
# ---------------------------------------------------------------------------

def test_extract_latex_returns_empty_string_when_pix2tex_raises():
    """extract_latex returns '' when pix2tex raises an exception."""
    import board_reader.ocr as ocr_module

    with patch.object(ocr_module, "_get_latex_ocr", side_effect=Exception("pix2tex failed")):
        result = extract_latex(np.zeros((10, 10, 3), dtype=np.uint8))

    assert result == ""


def test_extract_latex_returns_empty_string_when_model_returns_none():
    """extract_latex returns '' when the model returns None."""
    import board_reader.ocr as ocr_module

    mock_model = MagicMock()
    mock_model.return_value = None

    with patch.object(ocr_module, "_get_latex_ocr", return_value=mock_model):
        result = extract_latex(np.zeros((10, 10, 3), dtype=np.uint8))

    assert result == ""


def test_extract_latex_returns_str_on_success():
    """extract_latex returns the LaTeX string from the model."""
    import board_reader.ocr as ocr_module

    mock_model = MagicMock()
    mock_model.return_value = r"E = mc^2"

    with patch.object(ocr_module, "_get_latex_ocr", return_value=mock_model):
        result = extract_latex(np.zeros((10, 10, 3), dtype=np.uint8))

    assert isinstance(result, str)
    assert result == r"E = mc^2"
