"""Explanation client for IntelliAgent Board Reader.

Uses NVIDIA's OpenAI-compatible endpoint (mistral-small-24b-instruct) for
text-based board explanations, and the NIM vision model (llama-3.2-11b) for
free-form diagram descriptions.
"""

from __future__ import annotations

import base64
import logging
import time

import cv2
import numpy as np

from board_reader.config import Config
from board_reader.models import BoardState, ChangeDelta, StudentProfile

logger = logging.getLogger(__name__)

_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
_VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"


def build_gemini_prompt(
    delta: ChangeDelta,
    board_state: BoardState,
    config: Config,
    profile: StudentProfile | None = None,
    mode: str = "math",
) -> str:
    """Build the explanation prompt from the full board state.

    mode: "math" | "diagram" | "general" | "study"
    """
    grade_level = config.grade_level
    detail_line = (
        f"Explanation detail level: {profile.preferred_detail}\n" if profile is not None else ""
    )
    steps_text = "\n".join(f"  Step {s.id}: {s.text}" for s in board_state.board_steps)
    equations_text = "\n".join(f"  {eq}" for eq in board_state.equations) or "  (none)"

    if mode == "general":
        return (
            f"You are a helpful teacher explaining board content to a visually impaired {grade_level}th-grade student.\n"
            "The student cannot see the board — your spoken explanation is their only source of information.\n"
            "Do NOT use visual references such as \"as shown above\", \"see the diagram\", \"look at\", or \"as you can see\".\n"
            f"{detail_line}"
            "\n"
            f"Topic: {board_state.topic}\n"
            "\n"
            f"Board content:\n{steps_text}\n"
            "\n"
            f"Equations/formulas:\n{equations_text}\n"
            "\n"
            "Instructions:\n"
            f"- Describe and explain everything on the board clearly in spoken language for a {grade_level}th-grade student.\n"
            "- Cover all content shown — do not skip anything.\n"
            "- Read out any equations or formulas in plain spoken words.\n"
            "- Be thorough and clear. The student is relying entirely on your explanation."
        )

    if mode == "study":
        return (
            f"You are a Socratic tutor helping a visually impaired {grade_level}th-grade student understand board content.\n"
            "The student cannot see the board — your spoken explanation is their only source of information.\n"
            "Do NOT use visual references such as \"as shown above\", \"see the diagram\", \"look at\", or \"as you can see\".\n"
            f"{detail_line}"
            "\n"
            f"Topic: {board_state.topic}\n"
            "\n"
            f"Board content:\n{steps_text}\n"
            "\n"
            f"Equations/formulas:\n{equations_text}\n"
            "\n"
            "Instructions:\n"
            f"- First, briefly explain the topic and what is shown on the board in spoken language for a {grade_level}th-grade student.\n"
            "- Read out any equations in plain spoken words.\n"
            "- Then ask the student 2 to 3 short questions to test their understanding of the material.\n"
            "- Keep questions clear and specific to what was shown on the board.\n"
            "- Do not give away the answers in the questions."
        )

    # Default: math mode
    return (
        f"You are a kind, thorough math teacher explaining to a visually impaired {grade_level}th-grade student.\n"
        "The student cannot see the board — your spoken explanation is their only source of information.\n"
        "Do NOT use visual references such as \"as shown above\", \"see the diagram\", \"look at\", or \"as you can see\".\n"
        f"{detail_line}"
        "\n"
        f"Topic: {board_state.topic}\n"
        "\n"
        f"Board steps:\n{steps_text}\n"
        "\n"
        f"Equations:\n{equations_text}\n"
        "\n"
        "Instructions:\n"
        f"- Explain the full solution clearly and step by step in spoken language suitable for a {grade_level}th-grade student.\n"
        "- For each step, explain WHAT is being done and WHY.\n"
        "- Read out all equations in plain spoken words (e.g. 'x equals negative four plus or minus the square root of sixteen minus sixteen, all divided by two').\n"
        "- Be thorough — do not skip steps. The student is relying entirely on your explanation.\n"
        "- Do not introduce methods not shown on the board."
    )


def call_gemini_api(
    delta: ChangeDelta,
    board_state: BoardState,
    config: Config,
    mode: str = "math",
) -> str | None:
    """Generate a pedagogical explanation using Groq (fast) with NIM Mistral as fallback.

    mode: "math" | "general" | "study"
    Returns the explanation text string on success, or None on error.
    """
    prompt = build_gemini_prompt(delta, board_state, config, mode=mode)

    # Try Groq first if key is configured
    if config.groq_api_key:
        result = _call_groq(prompt, config)
        if result is not None:
            return result
        logger.warning("Groq failed — falling back to NIM Mistral")

    return _call_nim_mistral(prompt, config)


def _call_groq(prompt: str, config: Config) -> str | None:
    """Call Groq API (openai-compatible). Returns text or None on failure."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return None

    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=config.groq_api_key)
    for attempt in range(2):
        try:
            completion = client.chat.completions.create(
                model=config.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                top_p=0.9,
                max_tokens=2048,
                stream=True,
            )
            parts = [c.choices[0].delta.content for c in completion
                     if c.choices[0].delta.content is not None]
            text = "".join(parts)
            logger.info("Groq explain ok. length=%d chars model=%s", len(text), config.groq_model)
            return text
        except Exception as exc:
            exc_str = str(exc)
            if ("429" in exc_str or "rate" in exc_str.lower()) and attempt == 0:
                wait = _parse_retry_delay(exc_str)
                logger.warning("Groq rate limited, waiting %.0fs", wait)
                time.sleep(wait)
                continue
            logger.error("Groq API error: %s", exc)
            return None
    return None


def _call_nim_mistral(prompt: str, config: Config) -> str | None:
    """Call NIM Mistral as fallback. Returns text or None on failure."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        logger.error("openai package not available: %s", exc)
        return None

    client = OpenAI(base_url=_NIM_BASE_URL, api_key=config.nim_api_key)
    for attempt in range(2):
        try:
            completion = client.chat.completions.create(
                model=config.nim_explain_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                top_p=0.9,
                max_tokens=2048,
                stream=True,
            )
            parts = [c.choices[0].delta.content for c in completion
                     if c.choices[0].delta.content is not None]
            return "".join(parts)
        except Exception as exc:
            exc_str = str(exc)
            if ("429" in exc_str or "rate" in exc_str.lower()) and attempt == 0:
                wait = _parse_retry_delay(exc_str)
                logger.warning("NIM rate limited, waiting %.0fs", wait)
                time.sleep(wait)
                continue
            logger.error("NIM Mistral API error: %s", exc)
            return None
    return None


def explain_diagram(image: np.ndarray, config: Config) -> str | None:
    """Send an image to the NIM vision model for free-form diagram description.

    Bypasses the structured JSON schema — suitable for diagrams, charts, biology figures.
    Returns the explanation string on success, or None on failure.
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        logger.error("Explain client: openai package not available: %s", exc)
        return None

    # Encode image as PNG base64
    success, buf = cv2.imencode(".png", image)
    if not success:
        logger.error("Explain client: failed to encode image for diagram explanation")
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    grade_level = config.grade_level
    prompt_text = (
        f"You are a science and biology teacher for a visually impaired {grade_level}th-grade student.\n"
        "Do NOT use visual references such as \"as shown\", \"see the diagram\", \"look at\", or \"as you can see\".\n"
        "Describe this diagram in clear spoken language.\n"
        "Identify what the diagram shows, name and explain each labelled part, "
        "and describe how the parts relate to each other.\n"
        "Use simple, clear language suitable for a student who cannot see the image."
    )

    client = OpenAI(base_url=_NIM_BASE_URL, api_key=config.nim_api_key)

    for attempt in range(2):
        try:
            completion = client.chat.completions.create(
                model=_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ],
                temperature=0.2,
                top_p=0.7,
                max_tokens=1024,
                stream=True,
            )
            parts = []
            for chunk in completion:
                delta_content = chunk.choices[0].delta.content
                if delta_content is not None:
                    parts.append(delta_content)
            return "".join(parts)
        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "rate" in exc_str.lower():
                retry_wait = _parse_retry_delay(exc_str)
                if attempt == 0:
                    logger.warning(
                        "Explain client: rate limited, waiting %.0fs before retry", retry_wait
                    )
                    time.sleep(retry_wait)
                    continue
            logger.error("Explain client: diagram API error: %s", exc)
            return None

    return None


def _parse_retry_delay(error_str: str) -> float:
    """Extract retry delay in seconds from a rate-limit error string, default 60s."""
    import re
    match = re.search(r"retryDelay.*?(\d+)s", error_str)
    if match:
        return float(match.group(1))
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", error_str)
    if match:
        return float(match.group(1))
    return 60.0
