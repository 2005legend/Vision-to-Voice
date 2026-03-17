# Feature: intelliagent-board-reader, Property 1: Preprocessing produces grayscale output
"""Property-based tests for board_reader.capture.preprocess.

**Validates: Requirements 1.2**
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from board_reader.capture import preprocess


def _bgr_image_strategy():
    """Generate random BGR images of arbitrary size and pixel values."""
    height = st.integers(min_value=1, max_value=256)
    width = st.integers(min_value=1, max_value=256)

    @st.composite
    def _build(draw):
        h = draw(height)
        w = draw(width)
        pixels = draw(
            st.lists(
                st.integers(min_value=0, max_value=255),
                min_size=h * w * 3,
                max_size=h * w * 3,
            )
        )
        return np.array(pixels, dtype=np.uint8).reshape((h, w, 3))

    return _build()


# Feature: intelliagent-board-reader, Property 1: Preprocessing produces grayscale output
@given(frame=_bgr_image_strategy())
@settings(max_examples=100)
def test_preprocess_produces_grayscale_same_spatial_dims(frame):
    """For any valid BGR input image, preprocess returns a single-channel image
    with the same spatial dimensions (H, W)."""
    result = preprocess(frame)

    # Output must be 2-D (single channel)
    assert result.ndim == 2, f"Expected 2-D output, got shape {result.shape}"

    # Spatial dimensions must match input
    assert result.shape == frame.shape[:2], (
        f"Spatial dims mismatch: input {frame.shape[:2]}, output {result.shape}"
    )

    # Output dtype must be uint8
    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"
