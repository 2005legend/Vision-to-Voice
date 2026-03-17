"""Reinforcement learning behaviour tracker for IntelliAgent Board Reader."""

from __future__ import annotations

import json
import logging

from board_reader.models import StudentProfile

logger = logging.getLogger(__name__)


def record_feedback(profile: StudentProfile, event: str) -> None:
    """Legacy alias — kept for backward compatibility with existing tests."""
    record_event(profile, event)


def record_event(profile: StudentProfile, event: str) -> None:
    """Record a student behaviour event into the StudentProfile.

    event values:
      "explanation" — a new board explanation was spoken (heartbeat)
      "interrupt"   — student pressed Space to interrupt TTS
      "repeat"      — student asked to repeat / say again
      "followup"    — student asked a follow-up question
      "skip"        — legacy: maps to interrupt_count
      "replay"      — legacy: maps to repeat_count
    """
    if event == "explanation":
        profile.explanation_count += 1
    elif event == "interrupt":
        profile.interrupt_count += 1
    elif event == "repeat":
        profile.repeat_count += 1
    elif event == "followup":
        profile.followup_count += 1
    # Legacy events kept for backward compat
    elif event == "skip":
        profile.skip_count += 1
        profile.interrupt_count += 1
    elif event == "replay":
        profile.replay_count += 1
        profile.repeat_count += 1
    else:
        logger.warning("record_event: unknown event %r — ignoring", event)


def adapt_profile(profile: StudentProfile) -> StudentProfile:
    """Apply rule-based adaptation to preferred_detail.

    Rules only fire after at least 5 explanations to avoid adapting on noise.
    Counters reset after a rule fires so the next window starts fresh.
    """
    if profile.explanation_count < 5:
        return profile

    rule_fired = False

    if profile.interrupt_count > 3:
        profile.preferred_detail = "slow"
        logger.info("RL: high interrupts (%d) → preferred_detail=slow", profile.interrupt_count)
        rule_fired = True
    elif profile.repeat_count > 2:
        profile.preferred_detail = "brief"
        logger.info("RL: high repeats (%d) → preferred_detail=brief", profile.repeat_count)
        rule_fired = True
    elif profile.followup_count > 3:
        profile.preferred_detail = "detailed"
        logger.info("RL: high followups (%d) → preferred_detail=detailed", profile.followup_count)
        rule_fired = True

    if rule_fired:
        profile.interrupt_count = 0
        profile.repeat_count = 0
        profile.followup_count = 0
        profile.explanation_count = 0

    return profile


def check_in_needed(profile: StudentProfile, silence_threshold: float = 120.0) -> bool:
    """Return True if the student has been silent long enough to warrant a check-in."""
    return profile.silence_duration >= silence_threshold


def persist_profile(profile: StudentProfile, path: str) -> None:
    """Serialise StudentProfile to JSON and write to disk."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(profile.__dict__, fh, indent=2)
    logger.info("RL: profile persisted to %s", path)


def load_profile(path: str) -> StudentProfile:
    """Read JSON from disk and deserialise back to StudentProfile.

    Returns a default StudentProfile if the file does not exist.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Fill in any new fields that may not exist in older saved profiles
        defaults = StudentProfile.__dataclass_fields__
        for field_name, field_obj in defaults.items():
            if field_name not in data:
                data[field_name] = field_obj.default
        return StudentProfile(**{k: v for k, v in data.items() if k in defaults})
    except FileNotFoundError:
        logger.info("RL: no profile found at %s — starting fresh", path)
        return StudentProfile(grade_level=10)
    except Exception as exc:
        logger.warning("RL: failed to load profile from %s: %s — starting fresh", path, exc)
        return StudentProfile(grade_level=10)
