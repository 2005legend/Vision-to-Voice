# Feature: intelliagent-board-reader, Property 9: TTS queue preserves ordering under concurrent enqueue

from __future__ import annotations

import threading
import time
from unittest.mock import patch

from hypothesis import given, settings, strategies as st

from board_reader.tts_engine import TTSEngine


# ---------------------------------------------------------------------------
# Property 9: TTS queue preserves ordering under concurrent enqueue
# Validates: Requirements 6.2
# ---------------------------------------------------------------------------

@given(texts=st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=20))
@settings(max_examples=100)
def test_tts_queue_preserves_ordering_under_concurrent_enqueue(texts: list[str]) -> None:
    """Validates: Requirements 6.2

    For any sequence of strings enqueued while the TTS worker is busy,
    the strings SHALL be dequeued in the same order they were enqueued.
    """
    dequeued: list[str] = []
    processing_started = threading.Event()

    def fake_synthesise_and_play(text: str) -> None:
        """Mock synthesis: record dequeue order; first item stalls briefly."""
        dequeued.append(text)
        if not processing_started.is_set():
            processing_started.set()
            # Simulate the worker being "busy" so subsequent enqueues pile up
            time.sleep(0.05)

    engine = TTSEngine(model_name="tts_models/en/ljspeech/tacotron2-DDC")

    with patch.object(engine, "_synthesise_and_play", side_effect=fake_synthesise_and_play):
        engine.start()

        # Enqueue all items; the first will be picked up immediately and stall
        for text in texts:
            engine.enqueue(text)

        engine.stop(drain=True)

    assert dequeued == texts, (
        f"Dequeue order {dequeued!r} does not match enqueue order {texts!r}"
    )
