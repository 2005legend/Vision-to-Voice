# Feature: intelliagent-board-reader, Property 4: NIM request is correctly formed
# Feature: intelliagent-board-reader, Property 5: NIM JSON parsing round-trip

from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from board_reader.models import BoardState, BoardStep
from board_reader.nim_client import _MODEL, _build_payload, _encode_image, call_nim_api


# ---------------------------------------------------------------------------
# Helpers / strategies
# ---------------------------------------------------------------------------

def _make_config(endpoint="https://integrate.api.nvidia.com/v1/chat/completions", api_key="test-key"):
    cfg = MagicMock()
    cfg.nim_endpoint = endpoint
    cfg.nim_api_key = api_key
    cfg.nim_retry_wait = 0.0
    return cfg


def _small_image_strategy():
    """Generate small valid BGR images (1–8 pixels per side)."""
    return st.builds(
        lambda h, w: np.zeros((h, w, 3), dtype=np.uint8),
        h=st.integers(min_value=1, max_value=8),
        w=st.integers(min_value=1, max_value=8),
    )


# ---------------------------------------------------------------------------
# Property 4: NIM request is correctly formed
# Validates: Requirements 3.1, 3.2
# ---------------------------------------------------------------------------

@given(
    image=_small_image_strategy(),
    ocr_text=st.text(max_size=200),
)
@settings(max_examples=100)
def test_nim_request_is_correctly_formed(image, ocr_text):
    """Validates: Requirements 3.1, 3.2

    For any image and OCR text, the HTTP request constructed SHALL include:
    - correct endpoint URL
    - model name (meta/llama-3.2-11b-vision-instruct)
    - base64-encoded image payload
    - OCR text in user message
    """
    config = _make_config()
    captured_calls = []

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"topic": "t", "board_steps": [], "equations": []}'}}]
    }

    def fake_post(url, json=None, headers=None, timeout=None):
        captured_calls.append({"url": url, "payload": json, "headers": headers})
        return mock_response

    with patch("board_reader.nim_client.requests.post", side_effect=fake_post):
        call_nim_api(image, ocr_text, config)

    assert len(captured_calls) == 1, "Expected exactly one POST request"
    call = captured_calls[0]

    # Correct endpoint URL
    assert call["url"] == config.nim_endpoint

    # Authorization header present
    assert call["headers"]["Authorization"] == f"Bearer {config.nim_api_key}"

    payload = call["payload"]

    # Model name is correct
    assert payload["model"] == _MODEL

    # OCR text appears in user message
    user_message = payload["messages"][1]
    assert user_message["role"] == "user"
    content_parts = user_message["content"]
    text_parts = [p for p in content_parts if p["type"] == "text"]
    assert any(ocr_text in p["text"] for p in text_parts), (
        f"OCR text not found in user message text parts: {text_parts}"
    )

    # Base64-encoded image is present in image_url part
    image_parts = [p for p in content_parts if p["type"] == "image_url"]
    assert len(image_parts) == 1
    image_url = image_parts[0]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")

    # Verify the base64 payload is valid and non-empty
    b64_data = image_url[len("data:image/png;base64,"):]
    decoded = base64.b64decode(b64_data)
    assert len(decoded) > 0


# ---------------------------------------------------------------------------
# Property 5: NIM JSON parsing round-trip
# Validates: Requirements 3.3
# ---------------------------------------------------------------------------

@given(
    topic=st.text(),
    steps=st.lists(
        st.builds(BoardStep, id=st.integers(), text=st.text()),
        max_size=10,
    ),
    equations=st.lists(st.text(), max_size=10),
)
@settings(max_examples=100)
def test_nim_json_parsing_round_trip(topic, steps, equations):
    """Validates: Requirements 3.3

    For any JSON string conforming to the Board_State schema, parsing via
    BoardState.from_json SHALL produce a BoardState whose to_json() is
    semantically equivalent to the original JSON.
    """
    state = BoardState(topic=topic, board_steps=steps, equations=equations)
    original_json = state.to_json()

    # Parse back
    restored = BoardState.from_json(original_json)

    # to_json() of restored must be semantically equivalent (same JSON structure)
    restored_json = restored.to_json()

    assert json.loads(original_json) == json.loads(restored_json), (
        f"Round-trip mismatch:\n  original: {original_json}\n  restored: {restored_json}"
    )

    # Also verify equality via __eq__
    assert restored == state
