"""Config loader for IntelliAgent Board Reader."""

import sys
from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class Config:
    nim_api_key: str
    nim_explain_model: str     # text model for explanations, e.g. mistralai/mistral-small-24b-instruct
    camera_index: int          # default 0
    capture_interval: float    # seconds, default 2.0
    grade_level: int           # 10 or 12
    log_level: str             # DEBUG | INFO | WARNING | ERROR
    log_file: str              # path to rotating log file
    tts_model: str             # TTS model name
    nim_endpoint: str          # default https://integrate.api.nvidia.com/v1/chat/completions
    nim_retry_wait: float      # default 3.0
    rl_enabled: bool           # default false
    groq_api_key: str = ""     # Groq API key for fast text explanations
    groq_model: str = "llama-3.3-70b-versatile"
    change_threshold: float = 0.05   # pixel diff threshold for live camera loop
    hotkey: str = "space"            # hotkey to trigger STT interrupt
    stt_model: str = "base"          # faster-whisper model size
    rl_profile_path: str = "student_profile.json"
    # kept for backward-compat with tests that reference these fields
    gemini_api_key: str = ""
    gemini_model: str = ""


def _get_nested(data: dict, *keys: str) -> Any:
    """Traverse nested dict by keys; return None if any key is missing."""
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def load_config(path: str = "config.yaml") -> Config:
    """Load and validate config.yaml, returning a Config dataclass.

    Calls sys.exit(1) with a descriptive message on missing file or invalid value.
    """
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        sys.exit(f"Error: Configuration file '{path}' not found.")
    except yaml.YAMLError as e:
        sys.exit(f"Error: Failed to parse '{path}': {e}")

    if not isinstance(data, dict):
        sys.exit(f"Error: '{path}' must contain a YAML mapping at the top level.")

    def require(value: Any, key_path: str) -> Any:
        if value is None:
            sys.exit(f"Error: Required configuration key '{key_path}' is missing or empty in '{path}'.")
        return value

    nim_api_key = require(_get_nested(data, "nim", "api_key"), "nim.api_key")
    nim_explain_model = _get_nested(data, "nim", "explain_model")
    if nim_explain_model is None:
        nim_explain_model = "mistralai/mistral-small-24b-instruct"
    tts_model = require(_get_nested(data, "tts", "model"), "tts.model")
    log_level = require(_get_nested(data, "logging", "level"), "logging.level")
    log_file = require(_get_nested(data, "logging", "log_file"), "logging.log_file")
    grade_level = require(_get_nested(data, "student", "grade_level"), "student.grade_level")

    # Validate types / values
    valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR"}
    if log_level not in valid_log_levels:
        sys.exit(
            f"Error: 'logging.level' must be one of {sorted(valid_log_levels)}, got '{log_level}'."
        )

    if grade_level not in (10, 12):
        sys.exit(f"Error: 'student.grade_level' must be 10 or 12, got '{grade_level}'.")

    # Optional keys with defaults
    camera_index = _get_nested(data, "camera", "index")
    if camera_index is None:
        camera_index = 0
    camera_index = int(camera_index)

    capture_interval = _get_nested(data, "camera", "capture_interval")
    if capture_interval is None:
        capture_interval = 2.0
    capture_interval = float(capture_interval)

    nim_endpoint = _get_nested(data, "nim", "endpoint")
    if nim_endpoint is None:
        nim_endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"

    nim_retry_wait = _get_nested(data, "nim", "retry_wait")
    if nim_retry_wait is None:
        nim_retry_wait = 3.0
    nim_retry_wait = float(nim_retry_wait)

    rl_enabled = _get_nested(data, "rl", "enabled")
    if rl_enabled is None:
        rl_enabled = False
    rl_enabled = bool(rl_enabled)

    groq_api_key = _get_nested(data, "groq", "api_key") or ""
    groq_model = _get_nested(data, "groq", "model") or "llama-3.3-70b-versatile"

    change_threshold = _get_nested(data, "camera", "change_threshold")
    if change_threshold is None:
        change_threshold = 0.05
    change_threshold = float(change_threshold)

    hotkey = _get_nested(data, "hotkey") or "space"
    stt_model = _get_nested(data, "stt", "model") or "base"
    rl_profile_path = _get_nested(data, "rl", "profile_path") or "student_profile.json"

    return Config(
        nim_api_key=str(nim_api_key),
        nim_explain_model=str(nim_explain_model),
        camera_index=camera_index,
        capture_interval=capture_interval,
        grade_level=int(grade_level),
        log_level=str(log_level),
        log_file=str(log_file),
        tts_model=str(tts_model),
        nim_endpoint=str(nim_endpoint),
        nim_retry_wait=nim_retry_wait,
        rl_enabled=rl_enabled,
        groq_api_key=str(groq_api_key),
        groq_model=str(groq_model),
        change_threshold=change_threshold,
        hotkey=str(hotkey),
        stt_model=str(stt_model),
        rl_profile_path=str(rl_profile_path),
    )
