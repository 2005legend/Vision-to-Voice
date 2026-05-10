"""IntelliAgent orchestrator for IntelliAgent Board Reader."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from board_reader import gemini_client, nim_client
from board_reader.config import Config
from board_reader.models import BoardState, BoardStep, ChangeDelta, StudentProfile

if TYPE_CHECKING:
    from board_reader.tts_engine import TTSEngine

logger = logging.getLogger(__name__)


def call_nim(image: np.ndarray, ocr_text: str, config: Config) -> BoardState | None:
    """Call NIM VLM with a single image, parse into BoardState."""
    return call_nim_multi([image], ocr_text, config)


def call_nim_multi(images: list[np.ndarray], ocr_text: str, config: Config) -> BoardState | None:
    """Call NIM VLM with one or more images (multi-page), parse into BoardState."""
    raw = nim_client.call_nim_api_multi(images, ocr_text, config)
    if raw is None:
        return None
    try:
        topic = raw["topic"]
        board_steps = [BoardStep(id=s["id"], text=s["text"]) for s in raw["board_steps"]]
        equations = list(raw["equations"])
        return BoardState(topic=topic, board_steps=board_steps, equations=equations)
    except (KeyError, TypeError, ValueError) as exc:
        logger.error("IntelliAgent: malformed NIM response dict: %s | raw=%s", exc, raw)
        return None


def detect_change(current: BoardState, previous: BoardState | None) -> ChangeDelta | None:
    """Compare current BoardState against previous.

    Returns None iff current == previous.
    If previous is None, returns a ChangeDelta with all content as "added".
    Otherwise computes added_steps, changed_topic, and added_equations.
    """
    if previous is None:
        return ChangeDelta(
            added_steps=list(current.board_steps),
            changed_topic=current.topic,
            added_equations=list(current.equations),
        )

    if current == previous:
        return None

    prev_step_ids = {s.id for s in previous.board_steps}
    added_steps = [s for s in current.board_steps if s.id not in prev_step_ids]

    changed_topic = current.topic if current.topic != previous.topic else None

    prev_equations = set(previous.equations)
    added_equations = [eq for eq in current.equations if eq not in prev_equations]

    return ChangeDelta(
        added_steps=added_steps,
        changed_topic=changed_topic,
        added_equations=added_equations,
    )


def call_gemini(delta: ChangeDelta, current: BoardState, config: Config, profile: StudentProfile | None = None) -> str | None:
    """Delegate to gemini_client to generate a pedagogical explanation."""
    return gemini_client.call_gemini_api(delta, current, config, profile=profile)


def process_frame(
    image: np.ndarray,
    ocr_text: str,
    previous_board_state: BoardState | None,
    config: Config,
    tts_engine: TTSEngine,
    profile: StudentProfile | None = None,
) -> BoardState | None:
    """Run the per-frame pipeline.

    1. Call NIM to get current_state.
    2. If current_state is None, return previous_board_state unchanged.
    3. Detect change between current_state and previous_board_state.
    4. If delta is not None, call Gemini for an explanation.
    5. If explanation is not None, enqueue it to TTS.
    6. Return current_state.
    """
    current_state = call_nim(image, ocr_text, config)
    if current_state is None:
        return previous_board_state

    delta = detect_change(current_state, previous_board_state)
    if delta is not None:
        explanation = call_gemini(delta, current_state, config, profile=profile)
        if explanation is not None:
            tts_engine.enqueue(explanation)

    return current_state
