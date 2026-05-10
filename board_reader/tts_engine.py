"""TTS Engine — uses Windows SAPI via PowerShell for reliable repeated playback.

pyttsx3 on Windows returns a singleton engine; calling stop() kills the driver
permanently so the 2nd utterance never plays. We bypass this entirely by
spawning a PowerShell one-liner per utterance — zero state, always works.
"""

from __future__ import annotations

import logging
import queue
import re
import subprocess
import threading
import traceback

logger = logging.getLogger(__name__)

# Max characters per PowerShell SAPI call — avoids command-line length limits
_CHUNK_SIZE = 300


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-sized chunks for reliable TTS delivery."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: list[str] = []
    current = ""
    for s in sentences:
        if not s.strip():
            continue
        if len(current) + len(s) + 1 <= _CHUNK_SIZE:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            if len(s) > _CHUNK_SIZE:
                parts = re.split(r'(?<=,)\s+', s)
                sub = ""
                for p in parts:
                    if len(sub) + len(p) + 1 <= _CHUNK_SIZE:
                        sub = (sub + " " + p).strip()
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = p
                if sub:
                    chunks.append(sub)
            else:
                current = s
    if current:
        chunks.append(current)
    return chunks or [text]


def _speak_powershell(text: str) -> subprocess.Popen | None:
    """Start a PowerShell SAPI process. Returns Popen so caller can kill it."""
    safe = text.replace("'", "''")
    cmd = (
        f"Add-Type -AssemblyName System.Speech; "
        f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Rate = 1; "
        f"$s.Speak('{safe}')"
    )
    try:
        proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return proc
    except FileNotFoundError:
        _speak_pyttsx3(text)
        return None


def _speak_pyttsx3(text: str) -> None:
    """Fallback TTS via pyttsx3."""
    import pyttsx3  # type: ignore
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(text)
    engine.runAndWait()


class TTSEngine:
    """Queues text and plays it sequentially in a background thread.

    Key design:
    - _current_proc: the active PowerShell process (None when idle)
    - _proc_lock: protects _current_proc across threads
    - interrupt(): kills _current_proc immediately + drains queue + unblocks worker
    - pause() / resume(): pause after current chunk (used for STT barge-in)
    """

    def __init__(self, model_name: str = "") -> None:
        self.model_name = model_name
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()          # set = not paused
        self._thread: threading.Thread | None = None
        self._current_proc: subprocess.Popen | None = None
        self._proc_lock = threading.Lock()

    def enqueue(self, text: str) -> None:
        """Split text into sentence chunks and add to playback queue."""
        chunks = _split_sentences(text)
        logger.debug("TTS enqueue: %d chunk(s) from %d chars", len(chunks), len(text))
        for chunk in chunks:
            self._queue.put(chunk)

    def interrupt(self) -> None:
        """Instantly stop current speech and clear the queue.

        1. Kill the active PowerShell process (stops audio mid-word)
        2. Drain the queue (no more chunks will play)
        3. Ensure pause_event is SET so the worker can speak again after
        """
        logger.info("TTS: interrupt — killing active proc and clearing queue")

        # Kill the currently speaking process immediately
        with self._proc_lock:
            if self._current_proc is not None:
                try:
                    self._current_proc.kill()
                    logger.debug("TTS: killed active PowerShell proc")
                except Exception as e:
                    logger.debug("TTS: could not kill proc: %s", e)
                self._current_proc = None

        # Drain the queue
        drained = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                drained += 1
            except queue.Empty:
                break
        logger.debug("TTS: drained %d queued chunks", drained)

        # IMPORTANT: unblock the worker so it can speak again after interrupt
        self._pause_event.set()

    def pause(self) -> None:
        """Pause after the current chunk finishes (soft pause for STT)."""
        self._pause_event.clear()
        logger.debug("TTS: paused")

    def resume(self) -> None:
        """Resume speaking from the queue."""
        self._pause_event.set()
        logger.debug("TTS: resumed")

    def start(self) -> None:
        """Start the background worker thread."""
        self._stop_event.clear()
        self._pause_event.set()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="tts-worker")
        self._thread.start()
        logger.info("TTS worker started")

    def stop(self, drain: bool = True) -> None:
        """Stop the worker thread."""
        self.interrupt()  # kill any active speech first
        self._stop_event.set()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("TTS worker stopped")

    def _worker(self) -> None:
        logger.debug("TTS worker: running")
        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:
                self._queue.task_done()
                break

            # Wait if paused — unblocks instantly on resume() or interrupt()
            self._pause_event.wait()

            # Check again after unblocking — interrupt() may have drained queue
            if self._stop_event.is_set():
                self._queue.task_done()
                break

            try:
                logger.info("TTS speaking: %r", text[:80])
                proc = _speak_powershell(text)

                if proc is not None:
                    with self._proc_lock:
                        self._current_proc = proc
                    try:
                        proc.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    finally:
                        with self._proc_lock:
                            # Only clear if it's still our proc (interrupt may have set it to None)
                            if self._current_proc is proc:
                                self._current_proc = None

                logger.info("TTS done speaking chunk")
            except Exception as exc:
                logger.error("TTS error: %s\n%s", exc, traceback.format_exc())
            finally:
                self._queue.task_done()

        logger.debug("TTS worker: exited")
