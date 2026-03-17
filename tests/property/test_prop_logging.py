# Feature: intelliagent-board-reader, Property 11: Log entries contain required fields
"""Property-based tests for the logger module.

Validates: Requirements 9.1, 9.2
"""

import logging
import os
import re
import tempfile

from hypothesis import given, settings, strategies as st

from board_reader.config import Config
from board_reader.logger import get_logger

# Timestamp pattern produced by %(asctime)s  e.g. "2024-01-15 09:30:00,123"
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+")

# Valid stage names: non-empty strings that are valid Python logger names
_stage_name_st = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_"),
    min_size=1,
    max_size=40,
)

# Exclude carriage returns (\r) because Python's text-mode file reading
# translates standalone \r to \n (universal newlines), so the exact byte
# sequence would not survive a write-then-read round-trip in text mode.
_message_st = st.text(min_size=1, max_size=200).filter(lambda s: "\r" not in s)


def _make_config(log_file: str, log_level: str = "DEBUG") -> Config:
    return Config(
        nim_api_key="test-key",
        nim_explain_model="mistralai/mistral-small-24b-instruct",
        camera_index=0,
        capture_interval=2.0,
        grade_level=10,
        log_level=log_level,
        log_file=log_file,
        tts_model="tts_models/en/ljspeech/tacotron2-DDC",
        nim_endpoint="https://integrate.api.nvidia.com/v1/chat/completions",
        nim_retry_wait=3.0,
        rl_enabled=False,
    )


@given(stage_name=_stage_name_st, message=_message_st)
@settings(max_examples=100)
def test_log_entry_contains_stage_name_and_timestamp(stage_name: str, message: str) -> None:
    """For any pipeline stage name and log message, the emitted log entry SHALL
    contain a timestamp and the stage name.

    **Validates: Requirements 9.1, 9.2**
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as tmp:
        log_file = tmp.name

    try:
        config = _make_config(log_file)
        # Use a unique logger name to avoid handler pollution across examples
        unique_name = f"{stage_name}_{id(config)}"
        logger = get_logger(unique_name, config)
        logger.info(message)

        # Flush handlers so the entry is written to disk
        for handler in logger.handlers:
            handler.flush()

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        # The stage name (unique_name contains stage_name as prefix) must appear
        assert stage_name in content, (
            f"Stage name '{stage_name}' not found in log entry: {content!r}"
        )
        # A timestamp must be present
        assert _TIMESTAMP_RE.search(content), (
            f"No timestamp pattern found in log entry: {content!r}"
        )
    finally:
        # Clean up logger handlers to avoid file-handle leaks across examples
        logger = logging.getLogger(unique_name)
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        if os.path.exists(log_file):
            os.unlink(log_file)


@given(stage_name=_stage_name_st, error_message=_message_st)
@settings(max_examples=100)
def test_error_log_entry_contains_error_message(stage_name: str, error_message: str) -> None:
    """When an error occurs, the log entry SHALL contain the error message.

    **Validates: Requirements 9.1, 9.2**
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as tmp:
        log_file = tmp.name

    try:
        config = _make_config(log_file)
        unique_name = f"{stage_name}_err_{id(config)}"
        logger = get_logger(unique_name, config)
        logger.error(error_message)

        for handler in logger.handlers:
            handler.flush()

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert error_message in content, (
            f"Error message '{error_message}' not found in log entry: {content!r}"
        )
        assert _TIMESTAMP_RE.search(content), (
            f"No timestamp pattern found in log entry: {content!r}"
        )
    finally:
        logger = logging.getLogger(unique_name)
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        if os.path.exists(log_file):
            os.unlink(log_file)
