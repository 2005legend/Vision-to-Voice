"""STT Engine — speech-to-text for student doubt capture.

Primary: faster-whisper (local, offline, accurate)
Fallback: SpeechRecognition with Google Web Speech API
"""

from __future__ import annotations

import logging
import os
import tempfile
import wave
import typing

import numpy as np

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_CHUNK_SECONDS = 0.1          # 100ms chunks
_SILENCE_THRESHOLD = 0.005    # minimum ambient floor — quiet room baseline
_SILENCE_DURATION = 2.5       # seconds of silence before stopping


class STTEngine:
    """Records from mic and transcribes speech to text."""

    def __init__(self, model_size: str = "base") -> None:
        self._model_size = model_size
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore
            # Try CUDA first (RTX 3050), fall back to CPU
            try:
                self._model = WhisperModel(
                    self._model_size, device="cuda", compute_type="float16"
                )
                device_used = "cuda/float16"
            except Exception:
                self._model = WhisperModel(
                    self._model_size, device="cpu", compute_type="int8"
                )
                device_used = "cpu/int8"
            logger.info("STT: faster-whisper loaded (model=%s device=%s)", self._model_size, device_used)
            print(f"[STT] Using model: faster-whisper/{self._model_size} on {device_used}")
        except Exception as exc:
            logger.warning("STT: faster-whisper unavailable: %s - will use SpeechRecognition fallback", exc)
            print(f"[STT] faster-whisper failed to load: {exc} - falling back to SpeechRecognition")
            self._model = None

    def listen(self, timeout: float = 10.0, on_speech_start: typing.Callable[[], None] | None = None) -> str | None:
        """Record from mic until silence or timeout. Returns transcript or None.

        Never raises — all errors are caught and logged.
        """
        try:
            if self._model is not None:
                return self._listen_whisper(timeout, on_speech_start)
            return self._listen_fallback(timeout)
        except Exception as exc:
            logger.error("STT: listen() failed: %s", exc)
            return None

    def _listen_whisper(self, timeout: float, on_speech_start=None) -> str | None:
        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            logger.warning("STT: sounddevice not installed, falling back to SpeechRecognition")
            return self._listen_fallback(timeout)

        chunk_size = int(_SAMPLE_RATE * _CHUNK_SECONDS)
        max_chunks = int(timeout / _CHUNK_SECONDS)
        silence_chunks_needed = int(_SILENCE_DURATION / _CHUNK_SECONDS)

        frames: list[np.ndarray] = []
        silent_chunks = 0
        consecutive_voice = 0
        barge_in_triggered = False

        # Calibrate silence threshold from actual ambient noise (0.5s sample)
        calibration_chunks = 5
        ambient_rms_values: list[float] = []
        try:
            with sd.InputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="float32") as stream:
                for _ in range(calibration_chunks):
                    chunk, _ = stream.read(chunk_size)
                    ambient_rms_values.append(float(np.sqrt(np.mean(chunk ** 2))))
        except Exception:
            ambient_rms_values = [_SILENCE_THRESHOLD]
        ambient_rms = max(np.mean(ambient_rms_values), _SILENCE_THRESHOLD)
        # Multiplier 1.8: low enough to not cut speech, high enough to beat ambient noise
        dynamic_threshold = ambient_rms * 1.8
        print(f"[STT] Ambient RMS: {ambient_rms:.4f} → silence threshold: {dynamic_threshold:.4f}")
        logger.info("STT: ambient=%.4f threshold=%.4f", ambient_rms, dynamic_threshold)

        logger.info("STT: listening (timeout=%.1fs)…", timeout)
        print("[STT] Listening...")
        try:
            with sd.InputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="float32") as stream:
                for _ in range(max_chunks):
                    chunk, _ = stream.read(chunk_size)
                    frames.append(chunk.copy())
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    if rms < dynamic_threshold:
                        silent_chunks += 1
                        consecutive_voice = 0
                        if silent_chunks >= silence_chunks_needed and len(frames) > 10:
                            break
                    else:
                        silent_chunks = 0
                        consecutive_voice += 1
                        if on_speech_start and not barge_in_triggered and consecutive_voice >= 3:
                            logger.info("STT: Voice barge-in detected!")
                            barge_in_triggered = True
                            on_speech_start()
        except Exception as exc:
            logger.error("STT: sounddevice recording failed: %s", exc)
            return self._listen_fallback(timeout)

        if len(frames) < 5:
            logger.debug("STT: too few frames recorded — returning None")
            return None

        audio = np.concatenate(frames, axis=0).flatten()
        return self._transcribe_whisper(audio)

    def _transcribe_whisper(self, audio: np.ndarray) -> str | None:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(_SAMPLE_RATE)
                wf.writeframes((audio * 32767).astype(np.int16).tobytes())
            print(f"[STT] Audio duration: {len(audio) / _SAMPLE_RATE:.1f}s")
            segments, _ = self._model.transcribe(tmp_path, beam_size=5)
            text = " ".join(seg.text.strip() for seg in segments).strip()
            print(f"[STT] Raw transcript: '{text}'")
            logger.info("STT: transcribed %d chars", len(text))
            return text if text else None
        except Exception as exc:
            logger.error("STT: whisper transcription failed: %s", exc)
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _listen_fallback(self, timeout: float) -> str | None:
        try:
            import speech_recognition as sr  # type: ignore
            r = sr.Recognizer()
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source, timeout=timeout, phrase_time_limit=timeout)
            text = r.recognize_google(audio)
            logger.info("STT: fallback transcribed %d chars", len(text))
            return text if text else None
        except ImportError:
            logger.debug("STT: SpeechRecognition not installed — no fallback available")
            return None
        except Exception as exc:
            # Only log as error if faster-whisper was also unavailable (total failure)
            # If faster-whisper is loaded, this path shouldn't be reached
            if self._model is None:
                logger.error("STT: fallback also failed: %s", exc)
            else:
                logger.debug("STT: fallback called unexpectedly: %s", exc)
            return None
