# Feature: intelliagent-board-reader, Property 8: Gemini prompt contains all required components

from __future__ import annotations

from unittest.mock import MagicMock

from hypothesis import given, settings, strategies as st

from board_reader.gemini_client import build_gemini_prompt
from board_reader.models import BoardState, BoardStep, ChangeDelta


# ---------------------------------------------------------------------------
# Helpers / strategies
# ---------------------------------------------------------------------------

def _make_config(grade_level: int) -> MagicMock:
    cfg = MagicMock()
    cfg.grade_level = grade_level
    cfg.gemini_api_key = "test-key"
    cfg.gemini_model = "gemini-2.0-flash"
    return cfg


board_step_strategy = st.builds(
    BoardStep,
    id=st.integers(min_value=0, max_value=1000),
    text=st.text(max_size=200),
)

board_state_strategy = st.builds(
    BoardState,
    topic=st.text(max_size=200),
    board_steps=st.lists(board_step_strategy, max_size=10),
    equations=st.lists(st.text(max_size=100), max_size=10),
)

change_delta_strategy = st.builds(
    ChangeDelta,
    added_steps=st.lists(board_step_strategy, max_size=5),
    changed_topic=st.one_of(st.none(), st.text(max_size=200)),
    added_equations=st.lists(st.text(max_size=100), max_size=5),
)

grade_level_strategy = st.sampled_from([10, 12])


# ---------------------------------------------------------------------------
# Property 8: Gemini prompt contains all required components
# Validates: Requirements 5.1, 5.2, 5.5
# ---------------------------------------------------------------------------

@given(
    delta=change_delta_strategy,
    board_state=board_state_strategy,
    grade_level=grade_level_strategy,
)
@settings(max_examples=100)
def test_gemini_prompt_contains_all_required_components(delta, board_state, grade_level):
    """Validates: Requirements 5.1, 5.2, 5.5

    For any ChangeDelta, BoardState, and grade level, the prompt constructed
    SHALL contain:
    - delta content (added steps and equations)
    - full current board state (as JSON)
    - grade level
    - instruction to avoid visual references
    """
    config = _make_config(grade_level)
    prompt = build_gemini_prompt(delta, board_state, config)

    # Grade level is present
    assert str(grade_level) in prompt, (
        f"Grade level '{grade_level}' not found in prompt"
    )

    # Full board state JSON is embedded
    board_json = board_state.to_json()
    assert board_json in prompt, (
        f"Full board state JSON not found in prompt"
    )

    # Board topic is present
    assert board_state.topic in prompt, (
        f"Board topic '{board_state.topic}' not found in prompt"
    )

    # Instruction to avoid visual references is present
    visual_ref_instruction = "Do NOT use visual references"
    assert visual_ref_instruction in prompt, (
        f"Visual reference avoidance instruction not found in prompt"
    )

    # At least one specific visual reference phrase is called out
    assert "as shown above" in prompt, (
        "'as shown above' not mentioned in visual reference instruction"
    )

    # Delta added_equations are represented in the prompt
    assert str(delta.added_equations) in prompt, (
        f"Delta added_equations not found in prompt"
    )

    # Delta added_steps texts are represented in the prompt
    step_texts = [s.text for s in delta.added_steps]
    assert str(step_texts) in prompt, (
        f"Delta added_steps texts not found in prompt"
    )


# ---------------------------------------------------------------------------
# Property 13: Gemini prompt reflects Student_Profile detail level
# Validates: Requirements 11.2
# ---------------------------------------------------------------------------

# Feature: intelliagent-board-reader, Property 13: Gemini prompt reflects Student_Profile detail level

from board_reader.models import StudentProfile

detail_level_strategy = st.sampled_from(["brief", "medium", "detailed"])

student_profile_strategy = st.builds(
    StudentProfile,
    grade_level=st.integers(min_value=1, max_value=12),
    skip_count=st.integers(min_value=0, max_value=1000),
    replay_count=st.integers(min_value=0, max_value=1000),
    preferred_detail=detail_level_strategy,
)


@given(
    delta=change_delta_strategy,
    board_state=board_state_strategy,
    grade_level=grade_level_strategy,
    profile=student_profile_strategy,
)
@settings(max_examples=100)
def test_gemini_prompt_reflects_student_profile_detail_level(
    delta, board_state, grade_level, profile
):
    """Validates: Requirements 11.2

    For any StudentProfile with a given preferred_detail value, the prompt
    parameters passed to Gemini SHALL match the detail level encoded in that
    profile.
    """
    config = _make_config(grade_level)
    prompt = build_gemini_prompt(delta, board_state, config, profile=profile)

    assert profile.preferred_detail in prompt, (
        f"preferred_detail '{profile.preferred_detail}' not found in prompt"
    )
    assert f"Explanation detail level: {profile.preferred_detail}" in prompt, (
        f"Detail level instruction not found in prompt for preferred_detail='{profile.preferred_detail}'"
    )
