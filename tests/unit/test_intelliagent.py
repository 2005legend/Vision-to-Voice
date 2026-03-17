"""Unit tests for board_reader.intelliagent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from board_reader.intelliagent import call_nim, detect_change, process_frame
from board_reader.models import BoardState, BoardStep, ChangeDelta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_image():
    return np.zeros((4, 4, 3), dtype=np.uint8)


def _make_config():
    cfg = MagicMock()
    cfg.grade_level = 10
    cfg.gemini_api_key = "test-key"
    cfg.gemini_model = "gemini-2.0-flash"
    cfg.nim_api_key = "test-nim-key"
    cfg.nim_endpoint = "https://example.com"
    cfg.nim_retry_wait = 0.0
    return cfg


def _state(topic="Math", steps=None, equations=None):
    return BoardState(
        topic=topic,
        board_steps=steps or [],
        equations=equations or [],
    )


# ---------------------------------------------------------------------------
# detect_change tests
# ---------------------------------------------------------------------------

def test_detect_change_returns_none_when_equal():
    """detect_change returns None when current == previous."""
    state = _state("Algebra", [BoardStep(1, "Step 1")], ["x^2"])
    assert detect_change(state, state) is None


def test_detect_change_returns_none_for_equal_copies():
    """detect_change returns None for two equal but distinct objects."""
    a = _state("Algebra", [BoardStep(1, "Step 1")], ["x^2"])
    b = _state("Algebra", [BoardStep(1, "Step 1")], ["x^2"])
    assert detect_change(a, b) is None


def test_detect_change_returns_delta_when_different():
    """detect_change returns a ChangeDelta when current != previous."""
    prev = _state("Algebra", [BoardStep(1, "Step 1")], [])
    curr = _state("Algebra", [BoardStep(1, "Step 1"), BoardStep(2, "Step 2")], [])
    result = detect_change(curr, prev)
    assert result is not None
    assert isinstance(result, ChangeDelta)


def test_detect_change_added_steps_by_id():
    """detect_change identifies added steps by id."""
    prev = _state(steps=[BoardStep(1, "Step 1")])
    curr = _state(steps=[BoardStep(1, "Step 1"), BoardStep(2, "Step 2")])
    delta = detect_change(curr, prev)
    assert delta is not None
    assert len(delta.added_steps) == 1
    assert delta.added_steps[0].id == 2


def test_detect_change_changed_topic():
    """detect_change captures topic change."""
    prev = _state(topic="Algebra")
    curr = _state(topic="Geometry")
    delta = detect_change(curr, prev)
    assert delta is not None
    assert delta.changed_topic == "Geometry"


def test_detect_change_unchanged_topic_is_none():
    """detect_change sets changed_topic to None when topic is unchanged."""
    prev = _state(topic="Algebra", equations=["x"])
    curr = _state(topic="Algebra", equations=["x", "y"])
    delta = detect_change(curr, prev)
    assert delta is not None
    assert delta.changed_topic is None


def test_detect_change_previous_none_returns_all_content():
    """detect_change with previous=None returns ChangeDelta with all content."""
    curr = _state("Calculus", [BoardStep(1, "Intro"), BoardStep(2, "Derive")], ["f'(x)"])
    delta = detect_change(curr, None)
    assert delta is not None
    assert len(delta.added_steps) == 2
    assert delta.changed_topic == "Calculus"
    assert delta.added_equations == ["f'(x)"]


def test_detect_change_previous_none_empty_state():
    """detect_change with previous=None and empty state returns delta with empty lists."""
    curr = _state("", [], [])
    delta = detect_change(curr, None)
    assert delta is not None
    assert delta.added_steps == []
    assert delta.changed_topic == ""
    assert delta.added_equations == []


# ---------------------------------------------------------------------------
# call_nim tests
# ---------------------------------------------------------------------------

def test_call_nim_returns_none_when_nim_client_returns_none():
    """call_nim returns None when nim_client.call_nim_api returns None."""
    config = _make_config()
    with patch("board_reader.intelliagent.nim_client.call_nim_api", return_value=None):
        result = call_nim(_blank_image(), "text", config)
    assert result is None


def test_call_nim_returns_none_for_malformed_dict_missing_keys():
    """call_nim returns None when nim_client returns a dict missing required keys."""
    config = _make_config()
    malformed = {"topic": "Math"}  # missing board_steps and equations
    with patch("board_reader.intelliagent.nim_client.call_nim_api", return_value=malformed):
        result = call_nim(_blank_image(), "text", config)
    assert result is None


def test_call_nim_returns_none_for_wrong_types():
    """call_nim returns None when nim_client returns a dict with wrong types."""
    config = _make_config()
    malformed = {"topic": "Math", "board_steps": "not-a-list", "equations": []}
    with patch("board_reader.intelliagent.nim_client.call_nim_api", return_value=malformed):
        result = call_nim(_blank_image(), "text", config)
    assert result is None


def test_call_nim_returns_board_state_for_valid_dict():
    """call_nim returns a BoardState when nim_client returns a valid dict."""
    config = _make_config()
    valid = {
        "topic": "Algebra",
        "board_steps": [{"id": 1, "text": "Step 1"}],
        "equations": ["x^2 + y^2 = r^2"],
    }
    with patch("board_reader.intelliagent.nim_client.call_nim_api", return_value=valid):
        result = call_nim(_blank_image(), "text", config)
    assert isinstance(result, BoardState)
    assert result.topic == "Algebra"
    assert len(result.board_steps) == 1
    assert result.board_steps[0].id == 1
    assert result.equations == ["x^2 + y^2 = r^2"]


def test_call_nim_returns_board_state_for_empty_valid_dict():
    """call_nim returns a BoardState for a minimal valid dict."""
    config = _make_config()
    valid = {"topic": "", "board_steps": [], "equations": []}
    with patch("board_reader.intelliagent.nim_client.call_nim_api", return_value=valid):
        result = call_nim(_blank_image(), "", config)
    assert isinstance(result, BoardState)
    assert result.topic == ""
    assert result.board_steps == []
    assert result.equations == []


# ---------------------------------------------------------------------------
# process_frame tests
# ---------------------------------------------------------------------------

def test_process_frame_returns_previous_when_nim_returns_none():
    """process_frame returns previous_board_state unchanged when NIM returns None."""
    config = _make_config()
    tts = MagicMock()
    previous = _state("Previous Topic")

    with patch("board_reader.intelliagent.call_nim", return_value=None):
        result = process_frame(_blank_image(), "text", previous, config, tts)

    assert result is previous
    tts.enqueue.assert_not_called()


def test_process_frame_no_tts_when_no_change():
    """process_frame does NOT enqueue TTS when no change is detected."""
    config = _make_config()
    tts = MagicMock()
    state = _state("Algebra", [BoardStep(1, "Step 1")], [])

    with (
        patch("board_reader.intelliagent.call_nim", return_value=state),
        patch("board_reader.intelliagent.detect_change", return_value=None),
    ):
        result = process_frame(_blank_image(), "text", state, config, tts)

    tts.enqueue.assert_not_called()
    assert result == state


def test_process_frame_enqueues_tts_when_change_and_gemini_returns_explanation():
    """process_frame enqueues TTS when change detected and Gemini returns explanation."""
    config = _make_config()
    tts = MagicMock()
    prev = _state("Algebra")
    curr = _state("Geometry")
    delta = ChangeDelta(added_steps=[], changed_topic="Geometry", added_equations=[])
    explanation = "The topic changed to Geometry."

    with (
        patch("board_reader.intelliagent.call_nim", return_value=curr),
        patch("board_reader.intelliagent.detect_change", return_value=delta),
        patch("board_reader.intelliagent.call_gemini", return_value=explanation),
    ):
        result = process_frame(_blank_image(), "text", prev, config, tts)

    tts.enqueue.assert_called_once_with(explanation)
    assert result == curr


def test_process_frame_no_tts_when_gemini_returns_none():
    """process_frame does NOT enqueue TTS when Gemini returns None."""
    config = _make_config()
    tts = MagicMock()
    prev = _state("Algebra")
    curr = _state("Geometry")
    delta = ChangeDelta(added_steps=[], changed_topic="Geometry", added_equations=[])

    with (
        patch("board_reader.intelliagent.call_nim", return_value=curr),
        patch("board_reader.intelliagent.detect_change", return_value=delta),
        patch("board_reader.intelliagent.call_gemini", return_value=None),
    ):
        result = process_frame(_blank_image(), "text", prev, config, tts)

    tts.enqueue.assert_not_called()
    assert result == curr


def test_process_frame_returns_current_state():
    """process_frame returns current_state (updates previous_board_state)."""
    config = _make_config()
    tts = MagicMock()
    prev = _state("Old")
    curr = _state("New")

    with (
        patch("board_reader.intelliagent.call_nim", return_value=curr),
        patch("board_reader.intelliagent.call_gemini", return_value=None),
    ):
        result = process_frame(_blank_image(), "text", prev, config, tts)

    assert result == curr
