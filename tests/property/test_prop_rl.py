# Feature: intelliagent-board-reader, Property 14: Student_Profile persistence round-trip

from __future__ import annotations

import os
import tempfile

from hypothesis import given, settings, strategies as st

from board_reader.models import StudentProfile
from board_reader.rl import load_profile, persist_profile


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

detail_strategy = st.sampled_from(["brief", "medium", "detailed"])

student_profile_strategy = st.builds(
    StudentProfile,
    grade_level=st.integers(min_value=1, max_value=12),
    skip_count=st.integers(min_value=0, max_value=10_000),
    replay_count=st.integers(min_value=0, max_value=10_000),
    preferred_detail=detail_strategy,
)


# ---------------------------------------------------------------------------
# Property 14: Student_Profile persistence round-trip
# Validates: Requirements 11.3
# ---------------------------------------------------------------------------

@given(profile=student_profile_strategy)
@settings(max_examples=100)
def test_student_profile_persistence_round_trip(profile: StudentProfile) -> None:
    """Validates: Requirements 11.3

    For any valid StudentProfile, serialising to disk and loading it back
    SHALL produce an object equal to the original.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        tmp_path = tmp.name

    try:
        persist_profile(profile, tmp_path)
        loaded = load_profile(tmp_path)
        assert loaded == profile, (
            f"Round-trip mismatch: original={profile!r}, loaded={loaded!r}"
        )
    finally:
        os.unlink(tmp_path)
