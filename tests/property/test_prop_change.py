# Feature: intelliagent-board-reader, Property 6: Change detection is consistent with equality
# Feature: intelliagent-board-reader, Property 7: Session state is updated after each frame

from __future__ import annotations

from unittest.mock import MagicMock

from hypothesis import given, settings, strategies as st

from board_reader.intelliagent import detect_change, process_frame
from board_reader.models import BoardState, BoardStep


# ---------------------------------------------------------------------------
# Helpers / strategies
# ---------------------------------------------------------------------------

board_step_strategy = st.builds(
    BoardStep,
    id=st.integers(min_value=0, max_value=100),
    text=st.text(max_size=100),
)

board_state_strategy = st.builds(
    BoardState,
    topic=st.text(max_size=100),
    board_steps=st.lists(board_step_strategy, max_size=5),
    equations=st.lists(st.text(max_size=50), max_size=5),
)


def _make_config():
    cfg = MagicMock()
    cfg.grade_level = 10
    cfg.gemini_api_key = "test-key"
    cfg.gemini_model = "gemini-2.0-flash"
    cfg.nim_api_key = "test-nim-key"
    cfg.nim_endpoint = "https://example.com"
    cfg.nim_retry_wait = 0.0
    return cfg


# ---------------------------------------------------------------------------
# Property 6: Change detection is consistent with equality
# Validates: Requirements 4.1, 4.2
# ---------------------------------------------------------------------------

@given(a=board_state_strategy, b=board_state_strategy)
@settings(max_examples=100)
def test_detect_change_consistent_with_equality(a: BoardState, b: BoardState):
    """Validates: Requirements 4.1, 4.2

    For any two BoardState objects a and b, detect_change(a, b) SHALL return
    None if and only if a == b.
    """
    result = detect_change(a, b)
    if a == b:
        assert result is None, (
            f"Expected None when a == b, but got {result}"
        )
    else:
        assert result is not None, (
            f"Expected a ChangeDelta when a != b, but got None"
        )


# ---------------------------------------------------------------------------
# Property 7: Session state is updated after each frame
# Validates: Requirements 4.4
# ---------------------------------------------------------------------------

@given(states=st.lists(board_state_strategy, min_size=1, max_size=10))
@settings(max_examples=100)
def test_session_state_updated_after_each_frame(states: list[BoardState]):
    """Validates: Requirements 4.4

    For any sequence of BoardState objects processed by process_frame, after
    processing frame n, the returned state SHALL equal the BoardState produced
    at frame n (or previous if NIM returned None).
    """
    config = _make_config()
    tts_engine = MagicMock()

    previous = None
    for expected_state in states:
        # Mock call_nim to return the expected state, and call_gemini to return None
        with (
            __import__("unittest.mock", fromlist=["patch"]).patch(
                "board_reader.intelliagent.call_nim", return_value=expected_state
            ),
            __import__("unittest.mock", fromlist=["patch"]).patch(
                "board_reader.intelliagent.call_gemini", return_value=None
            ),
        ):
            returned = process_frame(
                image=__import__("numpy").zeros((4, 4, 3), dtype=__import__("numpy").uint8),
                ocr_text="",
                previous_board_state=previous,
                config=config,
                tts_engine=tts_engine,
            )

        # After processing, returned state must equal the state produced at this frame
        assert returned == expected_state, (
            f"Expected returned state to equal {expected_state}, got {returned}"
        )
        previous = returned
