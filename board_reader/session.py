"""Session manager for IntelliAgent Board Reader."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List

import cv2

from board_reader.capture import capture_frame, preprocess
from board_reader.config import Config
from board_reader.intelliagent import process_frame
from board_reader.logger import get_logger
from board_reader.models import BoardState
from board_reader.ocr import combine_ocr, extract_latex, extract_text
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
        self.current_board_state: BoardState | None = None
        self.session_history: List[BoardState | None] = []
        self._logger = None

    def start(self, config: Config) -> None:
        """Initialise resources and run the frame loop.

        Args:
            config: Loaded :class:`Config` instance.
        """
        self._stop_flag = False

        # 1. Initialise logger
        self._logger = get_logger("session", config)

        # 2. Open camera
        self._cap = cv2.VideoCapture(config.camera_index)

        # 3. Create and start TTS engine
        self.tts_engine = TTSEngine(config.tts_model)
        self.tts_engine.start()

        # 4. Initialise session state
        self.current_board_state = None
        self.session_history = []

        # 5. Log session start time
        start_time = datetime.now(tz=timezone.utc).isoformat()
        self._logger.info("Session started at %s", start_time)

        # 6. Run the frame loop
        self._run_loop(config)

    def _run_loop(self, config: Config) -> None:
        """Main frame-processing loop; runs until _stop_flag is set."""
        while not self._stop_flag:
            # 1. Capture frame
            frame = capture_frame(config.camera_index)
            if frame is None:
                self._logger.error(
                    "capture_frame returned None; retrying in 5 seconds."
                )
                time.sleep(5)
                continue

            # 2. Preprocess
            preprocessed = preprocess(frame)

            # 3. OCR
            ocr_text = combine_ocr(
                extract_text(preprocessed),
                extract_latex(preprocessed),
            )

            # 4. Process frame through IntelliAgent pipeline
            self.current_board_state = process_frame(
                preprocessed,
                ocr_text,
                self.current_board_state,
                config,
                self.tts_engine,
            )

            # 5. Append to history
            self.session_history.append(self.current_board_state)

            # 6. Wait before next capture
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
