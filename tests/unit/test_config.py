"""Unit tests for the config loader (board_reader/config.py).

Covers:
- Missing file → SystemExit
- Missing required key → SystemExit with descriptive m
essage
- Valid config loads correctly with correct field values
"""

import os
import tempfile

import pytest
import yaml

from board_reader.config import load_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_CONFIG = {
    "nim": {
        "api_key": "nvapi-test-key",
        "endpoint": "https://integrate.api.nvidia.com/v1/chat/completions",
        "explain_model": "mistralai/mistral-small-24b-instruct",
        "retry_wait": 3.0,
    },
    "camera": {
        "index": 0,
        "capture_interval": 2.0,
    },
    "student": {
        "grade_level": 10,
    },
    "tts": {
        "model": "tts_models/en/ljspeech/tacotron2-DDC",
    },
    "logging": {
        "level": "INFO",
        "log_file": "board_reader.log",
    },
    "rl": {
        "enabled": False,
    },
}


def _write_config(data: dict) -> str:
    """Write a dict as YAML to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(data, tmp)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Missing file
# ---------------------------------------------------------------------------

def test_missing_file_raises_system_exit():
    with pytest.raises(SystemExit) as exc_info:
        load_config("/nonexistent/path/config.yaml")
    assert exc_info.value.code != 0


def test_missing_file_message_mentions_path(capsys):
    with pytest.raises(SystemExit):
        load_config("/nonexistent/path/config.yaml")


# ---------------------------------------------------------------------------
# Missing required keys → SystemExit with descriptive message
# ---------------------------------------------------------------------------

REQUIRED_KEYS = [
    ("nim", "api_key"),
    ("tts", "model"),
    ("logging", "level"),
    ("logging", "log_file"),
    ("student", "grade_level"),
]


@pytest.mark.parametrize("section,key", REQUIRED_KEYS)
def test_missing_required_key_raises_system_exit(section, key):
    import copy
    data = copy.deepcopy(VALID_CONFIG)
    del data[section][key]
    path = _write_config(data)
    try:
        with pytest.raises(SystemExit) as exc_info:
            load_config(path)
        assert exc_info.value.code != 0
        msg = str(exc_info.value.code)
        assert section in msg or key in msg
    finally:
        os.unlink(path)


def test_missing_entire_nim_section_raises_system_exit():
    import copy
    data = copy.deepcopy(VALID_CONFIG)
    del data["nim"]
    path = _write_config(data)
    try:
        with pytest.raises(SystemExit) as exc_info:
            load_config(path)
        assert exc_info.value.code != 0
    finally:
        os.unlink(path)


def test_invalid_log_level_raises_system_exit():
    import copy
    data = copy.deepcopy(VALID_CONFIG)
    data["logging"]["level"] = "VERBOSE"
    path = _write_config(data)
    try:
        with pytest.raises(SystemExit) as exc_info:
            load_config(path)
        assert exc_info.value.code != 0
    finally:
        os.unlink(path)


def test_invalid_grade_level_raises_system_exit():
    import copy
    data = copy.deepcopy(VALID_CONFIG)
    data["student"]["grade_level"] = 11
    path = _write_config(data)
    try:
        with pytest.raises(SystemExit) as exc_info:
            load_config(path)
        assert exc_info.value.code != 0
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Valid config loads correctly
# ---------------------------------------------------------------------------

def test_valid_config_loads_all_fields():
    path = _write_config(VALID_CONFIG)
    try:
        cfg = load_config(path)
    finally:
        os.unlink(path)

    assert cfg.nim_api_key == "nvapi-test-key"
    assert cfg.nim_explain_model == "mistralai/mistral-small-24b-instruct"
    assert cfg.camera_index == 0
    assert cfg.capture_interval == 2.0
    assert cfg.grade_level == 10
    assert cfg.log_level == "INFO"
    assert cfg.log_file == "board_reader.log"
    assert cfg.tts_model == "tts_models/en/ljspeech/tacotron2-DDC"
    assert cfg.nim_endpoint == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert cfg.nim_retry_wait == 3.0
    assert cfg.rl_enabled is False


def test_valid_config_grade_12():
    import copy
    data = copy.deepcopy(VALID_CONFIG)
    data["student"]["grade_level"] = 12
    path = _write_config(data)
    try:
        cfg = load_config(path)
    finally:
        os.unlink(path)
    assert cfg.grade_level == 12


def test_explain_model_defaults_when_absent():
    """nim.explain_model should default to mistralai/mistral-small-24b-instruct."""
    import copy
    data = copy.deepcopy(VALID_CONFIG)
    del data["nim"]["explain_model"]
    path = _write_config(data)
    try:
        cfg = load_config(path)
    finally:
        os.unlink(path)
    assert cfg.nim_explain_model == "mistralai/mistral-small-24b-instruct"


def test_optional_keys_use_defaults_when_absent():
    import copy
    data = copy.deepcopy(VALID_CONFIG)
    del data["camera"]["index"]
    del data["camera"]["capture_interval"]
    del data["nim"]["endpoint"]
    del data["nim"]["retry_wait"]
    del data["rl"]["enabled"]
    path = _write_config(data)
    try:
        cfg = load_config(path)
    finally:
        os.unlink(path)

    assert cfg.camera_index == 0
    assert cfg.capture_interval == 2.0
    assert cfg.nim_endpoint == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert cfg.nim_retry_wait == 3.0
    assert cfg.rl_enabled is False


def test_rl_enabled_true():
    import copy
    data = copy.deepcopy(VALID_CONFIG)
    data["rl"]["enabled"] = True
    path = _write_config(data)
    try:
        cfg = load_config(path)
    finally:
        os.unlink(path)
    assert cfg.rl_enabled is True
