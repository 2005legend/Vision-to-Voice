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
    # Split on sentence boundaries
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
            # If single sentence is too long, split on commas
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
    """Speak text via Windows SAPI through PowerShell. Returns the Popen object so it can be killed."""
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
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        return proc
    except FileNotFoundError:
        _speak_pyttsx3(text)
        return None


def _speak_pyttsx3(text: str) -> None:
    """Fallback: pyttsx3. Creates a fresh engine each call to avoid singleton issues."""
    import pyttsx3  # type: ignore
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(text)
    engine.runAndWait()


class TTSEngine:
    """Queues text and plays it sequentially in a background thread."""

    def __init__(self, model_name: str = "") -> None:
        self.model_name = model_name
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # set = not paused; clear = paused
        self._thread: threading.Thread | None = None
        self._current_proc: subprocess.Popen | None = None
        self._proc_lock = threading.Lock()

    def enqueue(self, text: str) -> None:
        """Split text into sentence chunks and add each to the playback queue."""
        chunks = _split_sentences(text)
        logger.debug("TTS enqueue: %d chunk(s) from %d chars", len(chunks), len(text))
        for chunk in chunks:
            self._queue.put(chunk)

    def interrupt(self) -> None:
        """Interrupt current speech — clear the queue and stop after current chunk."""
        # Drain the queue so no more chunks play
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except Exception:
                break
        self._pause_event.clear()
        logger.debug("TTS interrupted and queue cleared")

    def pause(self) -> None:
        """Signal worker to stop after the current chunk finishes."""
        self._pause_event.clear()
        logger.debug("TTS paused")

    def resume(self) -> None:
        """Allow worker to continue processing the queue."""
        self._pause_event.set()
        logger.debug("TTS resumed")
        
    def interrupt(self) -> None:
        """Instantly stop current speech and clear the queue (Voice Barge-in)."""
        logger.info("TTS interrupt triggered. Clearing queue...")
        # Empty the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
                
        # Instantly kill the currently playing PowerShell process
        with self._proc_lock:
            if self._current_proc:
                try:
                    self._current_proc.kill()
                    logger.debug("Killed active TTS process.")
                except Exception as e:
                    logger.debug(f"Could not kill TTS proc: {e}")
                self._current_proc = None

    def start(self) -> None:
        """Start the background worker thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="tts-worker")
        self._thread.start()
        logger.info("TTS worker started")

    def stop(self, drain: bool = True) -> None:
        """Stop the worker, optionally waiting for the queue to drain."""
        if drain:
            self._queue.join()
        self._stop_event.set()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=10)
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

            # Block here if paused — resumes instantly when resume() is called
            self._pause_event.wait()

            try:
                logger.info("TTS speaking: %r", text[:80])
                proc = _speak_powershell(text)
                
                if proc:
                    with self._proc_lock:
                        self._current_proc = proc
                    try:
                        # Wait for the process to finish speaking
                        proc.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    finally:
                        with self._proc_lock:
                            self._current_proc = None
                            
                logger.info("TTS done")
            except Exception as exc:
                logger.error("TTS error: %s\n%s", exc, traceback.format_exc())
            finally:
                self._queue.task_done()

        logger.debug("TTS worker: exited")
