# Feature: intelliagent-board-reader, Property 2: OCR pipeline functions always return strings
# Feature: intelliagent-board-reader, Property 3: Combined OCR output contains both components
"""Property-based tests for board_reader.ocr.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st


# ---------------------------------------------------------------------------
# Image generators
# ---------------------------------------------------------------------------

def _image_strategy():
    """Generate small uint8 images: blank, noise, or all-black."""
    height = st.integers(min_value=1, max_value=64)
    width = st.integers(min_value=1, max_value=64)

    @st.composite
    def _build(draw):
        h = draw(height)
        w = draw(width)
        kind = draw(st.sampled_from(["blank", "noise", "black"]))
        if kind == "blank":
            return np.full((h, w, 3), 255, dtype=np.uint8)
        elif kind == "black":
            return np.zeros((h, w, 3), dtype=np.uint8)
        else:  # noise
            pixels = draw(
                st.lists(
                    st.integers(min_value=0, max_value=255),
                    min_size=h * w * 3,
                    max_size=h * w * 3,
                )
            )
            return np.array(pixels, dtype=np.uint8).reshape((h, w, 3))

    return _build()


# ---------------------------------------------------------------------------
# Property 2: OCR pipeline functions always return strings
# ---------------------------------------------------------------------------

# Feature: intelliagent-board-reader, Property 2: OCR pipeline functions always return strings
@given(image=_image_strategy())
@settings(max_examples=100)
def test_extract_text_always_returns_str(image):
    """For any input image, extract_text SHALL return a str and NOT raise.

    **Validates: Requirements 2.1, 2.4**

    PaddleOCR is mocked so the test focuses on the error-handling contract,
    including the case where the underlying library raises an exception.
    """
    from board_reader import ocr as ocr_module

    # Case 1: library raises – must still return ""
    with patch.object(ocr_module, "_get_paddle_ocr", side_effect=RuntimeError("boom")):
        result = ocr_module.extract_text(image)
        assert isinstance(result, str)

    # Case 2: library returns a normal result
    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = [[[None, ("hello", 0.99)]]]
    with patch.object(ocr_module, "_get_paddle_ocr", return_value=mock_ocr):
        result = ocr_module.extract_text(image)
        assert isinstance(result, str)

    # Case 3: library returns None / empty
    mock_ocr2 = MagicMock()
    mock_ocr2.ocr.return_value = None
    with patch.object(ocr_module, "_get_paddle_ocr", return_value=mock_ocr2):
        result = ocr_module.extract_text(image)
        assert isinstance(result, str)


# Feature: intelliagent-board-reader, Property 2: OCR pipeline functions always return strings
@given(image=_image_strategy())
@settings(max_examples=100)
def test_extract_latex_always_returns_str(image):
    """For any input image, extract_latex SHALL return a str and NOT raise.

    **Validates: Requirements 2.2, 2.4**

    pix2tex is mocked so the test focuses on the error-handling contract,
    including the case where the underlying library raises an exception.
    """
    from board_reader import ocr as ocr_module

    # Case 1: library raises – must still return ""
    with patch.object(ocr_module, "_get_latex_ocr", side_effect=RuntimeError("boom")):
        result = ocr_module.extract_latex(image)
        assert isinstance(result, str)

    # Case 2: library returns a LaTeX string
    mock_model = MagicMock()
    mock_model.return_value = r"E = mc^2"
    with patch.object(ocr_module, "_get_latex_ocr", return_value=mock_model):
        # Also need PIL to work; patch it to avoid import issues
        mock_pil_image = MagicMock()
        with patch("board_reader.ocr.PILImage", create=True):
            # Use a simpler approach: patch the whole function body path
            pass
        result = ocr_module.extract_latex(image)
        assert isinstance(result, str)

    # Case 3: library returns None
    mock_model2 = MagicMock()
    mock_model2.return_value = None
    with patch.object(ocr_module, "_get_latex_ocr", return_value=mock_model2):
        result = ocr_module.extract_latex(image)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Property 3: Combined OCR output contains both components
# ---------------------------------------------------------------------------

# Feature: intelliagent-board-reader, Property 3: Combined OCR output contains both components
@given(
    text=st.text(),
    latex=st.text(),
)
@settings(max_examples=100)
def test_combine_ocr_contains_both_components(text: str, latex: str):
    """For any text t and LaTeX string l, combine_ocr(t, l) SHALL contain both
    t and l as substrings (when non-empty).

    **Validates: Requirements 2.3**
    """
    from board_reader.ocr import combine_ocr

    result = combine_ocr(text, latex)

    assert isinstance(result, str)

    if text:
        assert text in result, f"text {text!r} not found in combined output"
    if latex:
        assert latex in result, f"latex {latex!r} not found in combined output"

    # Both section headers must always be present
    assert "[TEXT]" in result
    assert "[LATEX]" in result
