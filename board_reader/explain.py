"""Single-image explain pipeline for IntelliAgent Board Reader.

Usage:
    python -m board_reader explain path/to/board.jpg
    python -m board_reader explain path/to/heart.jpg --mode diagram
"""

from __future__ import annotations

import sys

import cv2

from board_reader.capture import preprocess
from board_reader.config import Config
from board_reader.gemini_client import call_gemini_api, explain_diagram
from board_reader.intelliagent import call_nim, detect_change
from board_reader.logger import get_logger
from board_reader.models import BoardState
from board_reader.ocr import combine_ocr, extract_latex, extract_text
from board_reader.tts_engine import TTSEngine


def explain_image(image_path: str, config: Config, mode: str = "board") -> None:
    """Load an image, run the full pipeline, and speak the explanation.

    mode: "board" (default) — OCR + NIM + Gemini
          "diagram"         — Gemini vision only (no NIM)
    """
    logger = get_logger("explain", config)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: could not load image '{image_path}'")
        sys.exit(1)
    logger.info("Loaded image: %s  shape=%s", image_path, frame.shape)

    preprocessed = preprocess(frame)

    if mode == "diagram":
        print("Diagram mode — sending image directly to Gemini vision...")
        explanation = explain_diagram(preprocessed, config)
        if explanation is None:
            print("Gemini could not describe the diagram. Check board_reader.log for details.")
            sys.exit(1)
    else:
        # Board mode: OCR → NIM → Gemini
        print("Running OCR...")
        ocr_text = combine_ocr(extract_text(preprocessed), extract_latex(preprocessed))
        logger.debug("OCR result: %s", ocr_text[:200])

        print("Calling NIM VLM...")
        board_state: BoardState | None = call_nim(preprocessed, ocr_text, config)
        if board_state is None:
            print("NIM VLM could not parse the board. Check board_reader.log for details.")
            sys.exit(1)
        print(f"Board topic: {board_state.topic}")
        print(f"Steps: {len(board_state.board_steps)}  Equations: {len(board_state.equations)}")

        delta = detect_change(board_state, None)

        print("Generating explanation via Gemini...")
        explanation = call_gemini_api(delta, board_state, config)
        if explanation is None:
            print("Gemini could not generate an explanation. Check board_reader.log for details.")
            sys.exit(1)

    print("\n--- Explanation ---")
    print(explanation)
    print("-------------------\n")

    print("Speaking explanation...")
    tts = TTSEngine(config.tts_model)
    tts.start()
    tts.enqueue(explanation)
    tts.stop(drain=True)
    print("Done.")
