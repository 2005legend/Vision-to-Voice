"""Session manager for IntelliAgent Board Reader."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List

import cv2

from board_reader.capture import capture_frame, compute_frame_diff, preprocess
from board_reader.config import Config
from board_reader.doubt_handler import handle_doubt
from board_reader.intelliagent import process_frame
from board_reader.logger import get_logger
from board_reader.models import BoardState, StudentProfile
from board_reader.ocr import combine_ocr, extract_latex, extract_text
from board_reader.rl import adapt_profile, load_profile, persist_profile, record_event
from board_reader.stt_engine import STTEngine
from board_reader.tts_engine import TTSEngine


class SessionManager:
    """Manages the lifecycle of a board-reading session.

    Usage::

        manager = SessionManager()
        manager.start(config)   # blocks until stop() is called from another thread
        manager.stop()
    """

    def __init__(self) -> None:
        self._stop_flag: bool = False
        self._cap: cv2.VideoCapture | None = None
        self.tts_engine: TTSEngine | None = None
        self.stt_engine: STTEngine | None = None
        self.current_board_state: BoardState | None = None
        self.previous_raw_frame: cv2.VideoCapture | None = None
        self.session_history: List[BoardState | None] = []
        self._logger = None
        self._student_profile: StudentProfile | None = None
        self._hotkey_pressed: bool = False

    def start(self, config: Config) -> None:
        """Initialise resources and run the frame loop.

        Args:
            config: Loaded :class:`Config` instance.
        """
        self._stop_flag = False

        # 1. Initialise logger
        self._logger = get_logger("session", config)

        # 2. Load student profile
        self._student_profile = load_profile(config.rl_profile_path)
        self._logger.info("Student profile loaded: grade=%d, detail=%s",
                          self._student_profile.grade_level,
                          self._student_profile.preferred_detail)

        # 3. Open camera
        self._cap = cv2.VideoCapture(config.camera_index)

        # 4. Create and start TTS engine
        self.tts_engine = TTSEngine(config.tts_model)
        self.tts_engine.start()

        # 5. Create STT engine
        self.stt_engine = STTEngine(model_size=config.stt_model)
        self._logger.info("STT engine initialized: model=%s", config.stt_model)

        # 6. Initialise session state
        self.current_board_state = None
        self.previous_raw_frame = None
        self.session_history = []
        self._hotkey_pressed = False

        # 7. Log session start time
        start_time = datetime.now(tz=timezone.utc).isoformat()
        self._logger.info("Session started at %s", start_time)

        # 8. Setup hotkey listener
        self._setup_hotkey(config.hotkey)

        # 9. Run the frame loop
        self._run_loop(config)

    def _run_loop(self, config: Config) -> None:
        """Main frame-processing loop; runs until _stop_flag is set.

        Includes pixel-diff change detection gate to skip frames with no meaningful changes.
        """
        while not self._stop_flag:
            # 1. Capture frame
            frame = capture_frame(config.camera_index)
            if frame is None:
                self._logger.error(
                    "capture_frame returned None; retrying in 5 seconds."
                )
                time.sleep(5)
                continue

            # 2. Change detection gate
            diff = compute_frame_diff(frame, self.previous_raw_frame)
            if diff < config.change_threshold:
                self._logger.debug("Frame diff %.3f below threshold %.3f — skipping",
                                   diff, config.change_threshold)
                time.sleep(config.capture_interval)
                continue

            self._logger.info("Frame diff %.3f >= threshold %.3f — processing",
                              diff, config.change_threshold)
            self.previous_raw_frame = frame

            # 3. Preprocess
            preprocessed = preprocess(frame)

            # 4. OCR
            ocr_text = combine_ocr(
                extract_text(preprocessed),
                extract_latex(preprocessed),
            )

            # 5. Process frame through IntelliAgent pipeline
            new_board_state = process_frame(
                preprocessed,
                ocr_text,
                self.current_board_state,
                config,
                self.tts_engine,
                profile=self._student_profile,
            )

            # 6. Record RL event and adapt profile
            if self._student_profile and new_board_state is not None:
                record_event(self._student_profile, "explanation")
                self._student_profile = adapt_profile(self._student_profile)

            # 7. Update state and history
            self.current_board_state = new_board_state
            self.session_history.append(self.current_board_state)

            # 8. Wait before next capture
            time.sleep(config.capture_interval)

    def stop(self) -> None:
        """Signal the session to stop and release all resources.

        Drains the TTS queue before releasing the camera.
        """
        # 1. Set stop flag
        self._stop_flag = True

        # 2. Drain TTS queue
        if self.tts_engine is not None:
            self.tts_engine.stop(drain=True)

        # 3. Release camera
        if self._cap is not None:
            self._cap.release()

        # 4. Log session end time
        if self._logger is not None:
            end_time = datetime.now(tz=timezone.utc).isoformat()
            self._logger.info("Session stopped at %s", end_time)
