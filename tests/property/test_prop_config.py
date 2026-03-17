# Feature: intelliagent-board-reader, Property 10: Config loading round-trip
"""Property tests for config loader round-trip (Property 10).

Validates: Requirements 8.1
"""

import os
import tempfile

import yaml
from hypothesis import given, settings, strategies as st

from board_reader.config import load_config

_log_levels = st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR"])
_grade_levels = st.sampled_from([10, 12])
_nonempty_str = st.text(min_size=1, max_size=64).filter(lambda s: s.strip() != "")


@given(
    nim_api_key=_nonempty_str,
    nim_explain_model=_nonempty_str,
    camera_index=st.integers(min_value=0, max_value=10),
    capture_interval=st.floats(min_value=0.1, max_value=60.0, allow_nan=False, allow_infinity=False),
    grade_level=_grade_levels,
    log_level=_log_levels,
    log_file=_nonempty_str,
    tts_model=_nonempty_str,
    nim_endpoint=_nonempty_str,
    nim_retry_wait=st.floats(min_value=0.1, max_value=30.0, allow_nan=False, allow_infinity=False),
    rl_enabled=st.booleans(),
)
@settings(max_examples=100)
def test_config_round_trip(
    nim_api_key,
    nim_explain_model,
    camera_index,
    capture_interval,
    grade_level,
    log_level,
    log_file,
    tts_model,
    nim_endpoint,
    nim_retry_wait,
    rl_enabled,
):
    """For any valid configuration dict, serialise to a temp config.yaml and load
    via load_config → fields must match the original values exactly.

    Validates: Requirements 8.1
    """
    config_dict = {
        "nim": {
            "api_key": nim_api_key,
            "explain_model": nim_explain_model,
            "endpoint": nim_endpoint,
            "retry_wait": nim_retry_wait,
        },
        "camera": {
            "index": camera_index,
            "capture_interval": capture_interval,
        },
        "student": {
            "grade_level": grade_level,
        },
        "tts": {
            "model": tts_model,
        },
        "logging": {
            "level": log_level,
            "log_file": log_file,
        },
        "rl": {
            "enabled": rl_enabled,
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(config_dict, tmp)
        tmp_path = tmp.name

    try:
        cfg = load_config(tmp_path)
    finally:
        os.unlink(tmp_path)

    assert cfg.nim_api_key == nim_api_key
    assert cfg.nim_explain_model == nim_explain_model
    assert cfg.camera_index == camera_index
    assert abs(cfg.capture_interval - capture_interval) < 1e-9
    assert cfg.grade_level == grade_level
    assert cfg.log_level == log_level
    assert cfg.log_file == log_file
    assert cfg.tts_model == tts_model
    assert cfg.nim_endpoint == nim_endpoint
    assert abs(cfg.nim_retry_wait - nim_retry_wait) < 1e-9
    assert cfg.rl_enabled == rl_enabled
