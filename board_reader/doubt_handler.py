"""Doubt handler — answers student's spoken questions using current board context."""

from __future__ import annotations

import logging

from board_reader.config import Config
from board_reader.models import BoardState, StudentProfile

logger = logging.getLogger(__name__)

_DETAIL_HINTS: dict[str, str] = {
    "brief":    "Keep your answer concise — 2 to 3 sentences.",
    "detailed": "Explain thoroughly with sub-steps.",
    "slow":     "Use very short sentences. One idea per sentence.",
    "medium":   "Aim for 3 to 5 clear sentences.",
}

_FALLBACK_ANSWER = "Sorry, I couldn't answer that right now."


def build_doubt_prompt(
    question: str,
    board_state: BoardState,
    profile: StudentProfile,
) -> str:
    """Build the follow-up prompt for a student's spoken question."""
    steps_text = (
        "\n".join(f"  Step {s.id}: {s.text}" for s in board_state.board_steps)
        if board_state.board_steps
        else "  (no steps recorded)"
    )
    equations_text = ", ".join(board_state.equations) if board_state.equations else "none"
    detail_hint = _DETAIL_HINTS.get(profile.preferred_detail, _DETAIL_HINTS["medium"])

    return (
        f"Current board context:\n"
        f"  Topic: {board_state.topic}\n"
        f"  Steps:\n{steps_text}\n"
        f"  Equations: {equations_text}\n"
        f"\n"
        f'Student question: "{question}"\n'
        f"\n"
        f"Answer the student's question clearly in spoken language for a grade {profile.grade_level} student.\n"
        f"Do not use visual references like 'as shown' or 'see above'.\n"
        f"{detail_hint}"
    )


def handle_doubt(
    question: str,
    board_state: BoardState,
    config: Config,
    profile: StudentProfile,
) -> str:
    """Send student's spoken question + board context to Groq. Returns answer text.

    Never returns None — falls back to a polite error message on failure.
    """
    if not question or not question.strip():
        return _FALLBACK_ANSWER

    prompt = build_doubt_prompt(question, board_state, profile)

    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=config.groq_api_key)
        completion = client.chat.completions.create(
            model=config.groq_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a patient teacher answering a student's verbal question during class. "
                        "Be clear, concise, and use spoken language only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=300,
        )
        answer = (completion.choices[0].message.content or "").strip()
        logger.info("DoubtHandler: answered %d chars for question %r", len(answer), question[:60])
        return answer if answer else _FALLBACK_ANSWER
    except Exception as exc:
        logger.error("DoubtHandler: Groq error: %s", exc)
        return _FALLBACK_ANSWER
