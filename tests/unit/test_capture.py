"""Unit tests for board_reader.capture."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from board_reader.capture import capture_frame, preprocess


# ---------------------------------------------------------------------------
# preprocess tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("height,width", [(1, 1), (10, 10), (100, 200), (480, 640)])
def test_preprocess_output_shape_matches_input(height, width):
    """preprocess output (H, W) must match input (H, W)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    result = preprocess(frame)
    assert result.shape == (height, width)


def test_preprocess_output_dtype_is_uint8():
    """preprocess output dtype must be uint8."""
    frame = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    result = preprocess(frame)
    assert result.dtype == np.uint8


def test_preprocess_output_ndim_is_2():
    """preprocess output must be single-channel (ndim == 2)."""
    frame = np.random.randint(0, 256, (60, 80, 3), dtype=np.uint8)
    result = preprocess(frame)
    assert result.ndim == 2


# ---------------------------------------------------------------------------
# capture_frame error-handling tests
# ---------------------------------------------------------------------------

def test_capture_frame_returns_none_when_camera_not_opened():
    """capture_frame returns None when VideoCapture.isOpened() is False."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with patch("board_reader.capture.cv2.VideoCapture", return_value=mock_cap):
        result = capture_frame(0)

    assert result is None


def test_capture_frame_returns_none_when_read_fails():
    """capture_frame returns None when VideoCapture.read() returns (False, None)."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None)

    with patch("board_reader.capture.cv2.VideoCapture", return_value=mock_cap):
        result = capture_frame(0)

    assert result is None
