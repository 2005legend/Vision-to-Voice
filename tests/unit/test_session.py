"""Unit tests for board_reader.session.SessionManager."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch, call

import pytest

from board_reader.session import SessionManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(capture_interval: float = 0.0) -> MagicMock:
    cfg = MagicMock()
    cfg.camera_index = 0
    cfg.tts_model = "tts_models/en/ljspeech/tacotron2-DDC"
    cfg.capture_interval = capture_interval
    cfg.log_level = "INFO"
    cfg.log_file = "test_session.log"
    return cfg


def _blank_frame():
    import numpy as np
    return np.zeros((4, 4, 3), dtype=np.uint8)


def _blank_preprocessed():
    import numpy as np
    return np.zeros((4, 4), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Test: initial state after start (with mocked frame loop)
# ---------------------------------------------------------------------------

def test_initial_state_before_frame_loop():
    """After start is called (loop immediately stopped), initial state is correct."""
    manager = SessionManager()
    config = _make_config()

    # Patch everything heavy; make the loop exit immediately by setting stop flag
    # after the first iteration via a side-effect on capture_frame
    frame_count = 0

    def fake_capture(camera_index):
        nonlocal frame_count
        frame_count += 1
        manager._stop_flag = True  # stop after first attempt
        return None  # return None so we exercise the retry path without looping

    with (
        patch("board_reader.session.get_logger", return_value=MagicMock()),
        patch("board_reader.session.cv2.VideoCapture", return_value=MagicMock()),
        patch("board_reader.session.TTSEngine") as MockTTS,
        patch("board_reader.session.capture_frame", side_effect=fake_capture),
        patch("board_reader.session.time.sleep"),  # skip actual sleep
    ):
        mock_tts_instance = MagicMock()
        MockTTS.return_value = mock_tts_instance

        manager.start(config)

    # current_board_state starts as None (no successful frame processed)
    assert manager.current_board_state is None
    # session_history starts empty (no successful frame appended)
    assert manager.session_history == []


def test_session_history_populated_after_successful_frame():
    """session_history grows by one entry per successfully processed frame."""
    manager = SessionManager()
    config = _make_config()

    frame = _blank_frame()
    preprocessed = _blank_preprocessed()
    fake_board_state = MagicMock()

    # Use time.sleep side-effect to stop after the first frame is processed
    def fake_sleep(seconds):
        manager._stop_flag = True

    with (
        patch("board_reader.session.get_logger", return_value=MagicMock()),
        patch("board_reader.session.cv2.VideoCapture", return_value=MagicMock()),
        patch("board_reader.session.TTSEngine") as MockTTS,
        patch("board_reader.session.capture_frame", return_value=frame),
        patch("board_reader.session.preprocess", return_value=preprocessed),
        patch("board_reader.session.extract_text", return_value="text"),
        patch("board_reader.session.extract_latex", return_value=""),
        patch("board_reader.session.combine_ocr", return_value="[TEXT]\ntext\n[LATEX]\n"),
        patch("board_reader.session.process_frame", return_value=fake_board_state),
        patch("board_reader.session.time.sleep", side_effect=fake_sleep),
    ):
        mock_tts_instance = MagicMock()
        MockTTS.return_value = mock_tts_instance

        manager.start(config)

    assert len(manager.session_history) == 1
    assert manager.session_history[0] is fake_board_state
    assert manager.current_board_state is fake_board_state


# ---------------------------------------------------------------------------
# Test: stop() calls tts_engine.stop(drain=True)
# ---------------------------------------------------------------------------

def test_stop_drains_tts_queue():
    """stop() calls tts_engine.stop(drain=True) to drain the TTS queue."""
    manager = SessionManager()
    config = _make_config()

    mock_tts = MagicMock()

    with (
        patch("board_reader.session.get_logger", return_value=MagicMock()),
        patch("board_reader.session.cv2.VideoCapture", return_value=MagicMock()),
        patch("board_reader.session.TTSEngine", return_value=mock_tts),
        patch("board_reader.session.capture_frame", return_value=None),
        patch("board_reader.session.time.sleep", side_effect=lambda _: setattr(manager, "_stop_flag", True)),
    ):
        manager.start(config)

    manager.stop()

    mock_tts.stop.assert_called_once_with(drain=True)


def test_stop_releases_camera():
    """stop() calls release() on the camera capture object."""
    manager = SessionManager()
    config = _make_config()

    mock_cap = MagicMock()
    mock_tts = MagicMock()

    with (
        patch("board_reader.session.get_logger", return_value=MagicMock()),
        patch("board_reader.session.cv2.VideoCapture", return_value=mock_cap),
        patch("board_reader.session.TTSEngine", return_value=mock_tts),
        patch("board_reader.session.capture_frame", return_value=None),
        patch("board_reader.session.time.sleep", side_effect=lambda _: setattr(manager, "_stop_flag", True)),
    ):
        manager.start(config)

    manager.stop()

    mock_cap.release.assert_called_once()


# ---------------------------------------------------------------------------
# Test: camera error retry — None frame does NOT crash the session
# ---------------------------------------------------------------------------

def test_camera_error_does_not_crash_session():
    """When capture_frame returns None, the session retries without crashing."""
    manager = SessionManager()
    config = _make_config()

    none_count = 0

    def fake_capture(camera_index):
        nonlocal none_count
        none_count += 1
        if none_count >= 3:
            manager._stop_flag = True
        return None  # always return None to simulate camera failure

    sleep_calls = []

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    with (
        patch("board_reader.session.get_logger", return_value=MagicMock()),
        patch("board_reader.session.cv2.VideoCapture", return_value=MagicMock()),
        patch("board_reader.session.TTSEngine") as MockTTS,
        patch("board_reader.session.capture_frame", side_effect=fake_capture),
        patch("board_reader.session.time.sleep", side_effect=fake_sleep),
    ):
        MockTTS.return_value = MagicMock()
        # Should not raise
        manager.start(config)

    # Verify the 5-second retry sleep was called for each None frame
    retry_sleeps = [s for s in sleep_calls if s == 5]
    assert len(retry_sleeps) >= 2


def test_camera_error_retry_does_not_append_to_history():
    """When capture_frame returns None, nothing is appended to session_history."""
    manager = SessionManager()
    config = _make_config()

    call_count = 0

    def fake_capture(camera_index):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            manager._stop_flag = True
        return None

    with (
        patch("board_reader.session.get_logger", return_value=MagicMock()),
        patch("board_reader.session.cv2.VideoCapture", return_value=MagicMock()),
        patch("board_reader.session.TTSEngine") as MockTTS,
        patch("board_reader.session.capture_frame", side_effect=fake_capture),
        patch("board_reader.session.time.sleep"),
    ):
        MockTTS.return_value = MagicMock()
        manager.start(config)

    assert manager.session_history == []


# ---------------------------------------------------------------------------
# Test: stop() before start() does not crash
# ---------------------------------------------------------------------------

def test_stop_before_start_does_not_crash():
    """Calling stop() on a fresh SessionManager does not raise."""
    manager = SessionManager()
    manager.stop()  # should be a no-op


# ---------------------------------------------------------------------------
# Test: TTS engine is started during session start
# ---------------------------------------------------------------------------

def test_tts_engine_started_on_session_start():
    """TTSEngine.start() is called when the session starts."""
    manager = SessionManager()
    config = _make_config()

    mock_tts = MagicMock()

    with (
        patch("board_reader.session.get_logger", return_value=MagicMock()),
        patch("board_reader.session.cv2.VideoCapture", return_value=MagicMock()),
        patch("board_reader.session.TTSEngine", return_value=mock_tts),
        patch("board_reader.session.capture_frame", return_value=None),
        patch("board_reader.session.time.sleep", side_effect=lambda _: setattr(manager, "_stop_flag", True)),
    ):
        manager.start(config)

    mock_tts.start.assert_called_once()
