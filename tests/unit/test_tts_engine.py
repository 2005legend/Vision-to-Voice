"""Unit tests for board_reader.tts_engine.TTSEngine."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from board_reader.tts_engine import TTSEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine() -> TTSEngine:
    return TTSEngine(model_name="tts_models/en/ljspeech/tacotron2-DDC")


# ---------------------------------------------------------------------------
# Queue ordering
# ---------------------------------------------------------------------------

def test_queue_fifo_ordering():
    """Items enqueued are dequeued and processed in FIFO order."""
    dequeued: list[str] = []

    def fake_synth(text: str) -> None:
        dequeued.append(text)

    engine = _make_engine()
    with patch.object(engine, "_synthesise_and_play", side_effect=fake_synth):
        engine.start()
        for item in ["alpha", "beta", "gamma"]:
            engine.enqueue(item)
        engine.stop(drain=True)

    assert dequeued == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# Synthesis failure handling
# ---------------------------------------------------------------------------

def test_synthesis_failure_does_not_crash_engine():
    """When Coqui TTS raises, the engine logs the error and continues."""
    processed: list[str] = []

    call_count = 0

    def fake_synth(text: str) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("TTS model unavailable")
        processed.append(text)

    engine = _make_engine()
    with patch.object(engine, "_synthesise_and_play", side_effect=fake_synth):
        engine.start()
        engine.enqueue("fail-item")
        engine.enqueue("ok-item")
        engine.stop(drain=True)

    # Engine must not crash; the second item should still be processed
    assert processed == ["ok-item"]


# ---------------------------------------------------------------------------
# stop(drain=True)
# ---------------------------------------------------------------------------

def test_stop_with_drain_processes_all_items():
    """stop(drain=True) waits until every queued item has been processed."""
    processed: list[str] = []

    def fake_synth(text: str) -> None:
        time.sleep(0.01)  # simulate work
        processed.append(text)

    engine = _make_engine()
    items = [str(i) for i in range(5)]

    with patch.object(engine, "_synthesise_and_play", side_effect=fake_synth):
        engine.start()
        for item in items:
            engine.enqueue(item)
        engine.stop(drain=True)

    assert processed == items


# ---------------------------------------------------------------------------
# stop(drain=False)
# ---------------------------------------------------------------------------

def test_stop_without_drain_leaves_unprocessed_items():
    """stop(drain=False) stops the worker without draining remaining queued items.

    Per spec: in-progress playback is never interrupted, but items still in the
    queue (not yet dequeued) are abandoned when drain=False.
    """
    processed: list[str] = []
    first_started = threading.Event()

    def fake_synth(text: str) -> None:
        first_started.set()
        time.sleep(0.3)  # simulate work long enough that stop() fires mid-queue
        processed.append(text)

    engine = _make_engine()
    items = [str(i) for i in range(10)]

    with patch.object(engine, "_synthesise_and_play", side_effect=fake_synth):
        engine.start()
        for item in items:
            engine.enqueue(item)
        # Wait until the first item is being processed, then stop immediately
        first_started.wait(timeout=2.0)
        engine.stop(drain=False)

    # The first item was in-progress and must have completed (no interruption)
    assert "0" in processed
    # With drain=False and 0.3s per item, not all 10 items should be processed
    assert len(processed) < len(items)
