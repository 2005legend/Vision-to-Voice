"""Unit tests for board_reader.nim_client."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import requests

from board_reader.nim_client import call_nim_api


def _make_config(endpoint="https://integrate.api.nvidia.com/v1/chat/completions", api_key="test-key", retry_wait=0.0):
    cfg = MagicMock()
    cfg.nim_endpoint = endpoint
    cfg.nim_api_key = api_key
    cfg.nim_retry_wait = retry_wait
    return cfg


def _blank_image():
    return np.zeros((4, 4, 3), dtype=np.uint8)


def _ok_response(content_json: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "choices": [{"message": {"content": content_json}}]
    }
    return resp


# ---------------------------------------------------------------------------
# Retry logic tests
# ---------------------------------------------------------------------------

def test_retry_once_on_connection_error_then_returns_none():
    """On ConnectionError, exactly ONE retry is made, then returns None."""
    config = _make_config()
    post_calls = []

    def fake_post(*args, **kwargs):
        post_calls.append(1)
        raise requests.exceptions.ConnectionError("connection refused")

    with patch("board_reader.nim_client.requests.post", side_effect=fake_post):
        with patch("board_reader.nim_client.time.sleep"):
            result = call_nim_api(_blank_image(), "some text", config)

    assert result is None
    assert len(post_calls) == 2, f"Expected 2 attempts (1 + 1 retry), got {len(post_calls)}"


def test_retry_waits_configured_duration_on_network_error():
    """On network error, time.sleep is called with nim_retry_wait before retry."""
    config = _make_config(retry_wait=3.0)
    sleep_calls = []

    def fake_post(*args, **kwargs):
        raise requests.exceptions.ConnectionError("timeout")

    with patch("board_reader.nim_client.requests.post", side_effect=fake_post):
        with patch("board_reader.nim_client.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            call_nim_api(_blank_image(), "text", config)

    assert sleep_calls == [3.0]


def test_no_retry_on_malformed_json_response():
    """On HTTP 200 with invalid JSON body, no retry is made and None is returned."""
    config = _make_config()
    post_calls = []

    bad_resp = MagicMock()
    bad_resp.raise_for_status.return_value = None
    bad_resp.json.return_value = {
        "choices": [{"message": {"content": "this is not json {{{"}}]
    }

    def fake_post(*args, **kwargs):
        post_calls.append(1)
        return bad_resp

    with patch("board_reader.nim_client.requests.post", side_effect=fake_post):
        result = call_nim_api(_blank_image(), "text", config)

    assert result is None
    assert len(post_calls) == 1, f"Expected exactly 1 attempt (no retry), got {len(post_calls)}"


# ---------------------------------------------------------------------------
# Malformed JSON handling
# ---------------------------------------------------------------------------

def test_malformed_json_returns_none():
    """Returns None when the response content is not valid JSON."""
    config = _make_config()
    bad_resp = MagicMock()
    bad_resp.raise_for_status.return_value = None
    bad_resp.json.return_value = {"choices": [{"message": {"content": "not-json"}}]}

    with patch("board_reader.nim_client.requests.post", return_value=bad_resp):
        result = call_nim_api(_blank_image(), "ocr", config)

    assert result is None


def test_missing_choices_key_returns_none():
    """Returns None when response JSON is missing expected keys."""
    config = _make_config()
    bad_resp = MagicMock()
    bad_resp.raise_for_status.return_value = None
    bad_resp.json.return_value = {"unexpected": "structure"}

    with patch("board_reader.nim_client.requests.post", return_value=bad_resp):
        result = call_nim_api(_blank_image(), "ocr", config)

    assert result is None


# ---------------------------------------------------------------------------
# Successful response
# ---------------------------------------------------------------------------

def test_successful_response_returns_parsed_dict():
    """Returns the parsed dict when the response is valid JSON."""
    config = _make_config()
    expected = {"topic": "Algebra", "board_steps": [{"id": 1, "text": "Step 1"}], "equations": [r"x^2"]}
    resp = _ok_response('{"topic": "Algebra", "board_steps": [{"id": 1, "text": "Step 1"}], "equations": ["x^2"]}')

    with patch("board_reader.nim_client.requests.post", return_value=resp):
        result = call_nim_api(_blank_image(), "some ocr text", config)

    assert result == expected


def test_successful_response_empty_board():
    """Returns parsed dict for minimal valid board state."""
    config = _make_config()
    resp = _ok_response('{"topic": "", "board_steps": [], "equations": []}')

    with patch("board_reader.nim_client.requests.post", return_value=resp):
        result = call_nim_api(_blank_image(), "", config)

    assert result == {"topic": "", "board_steps": [], "equations": []}
