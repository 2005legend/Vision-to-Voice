"""Image capture and preprocessing for IntelliAgent Board Reader."""

import logging

import cv2
import numpy as np

logger = logging.getLogger("capture")


def capture_frame(camera_index: int) -> np.ndarray | None:
    """Capture a single frame from the camera.

    Returns the frame as a BGR ndarray, or None if the camera is unavailable
    or the read fails.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Camera index %d could not be opened.", camera_index)
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        logger.error("Failed to read frame from camera index %d.", camera_index)
        return None

    return frame


def compute_frame_diff(frame_a: np.ndarray | None, frame_b: np.ndarray | None) -> float:
    """Return fraction of pixels that changed between two raw BGR frames (0.0–1.0).

    Returns 1.0 if either frame is None (treat unknown as changed — always process).
    Handles frames of different sizes by resizing the smaller to match the larger.
    """
    if frame_a is None or frame_b is None:
        return 1.0

    if frame_a.shape != frame_b.shape:
        h = max(frame_a.shape[0], frame_b.shape[0])
        w = max(frame_a.shape[1], frame_b.shape[1])
        frame_a = cv2.resize(frame_a, (w, h))
        frame_b = cv2.resize(frame_b, (w, h))

    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_a, gray_b)
    changed_pixels = np.sum(diff > 25)  # ignore minor camera noise
    return float(changed_pixels) / float(gray_a.size)


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Preprocess a BGR frame for OCR and VLM input.

    Steps:
      1. BGR → grayscale
      2. Gaussian blur (noise reduction)
      3. CLAHE (contrast enhancement)
      4. Auto-invert if background is dark (white-on-black → black-on-white)
         PaddleOCR v2 expects dark text on light background.
      5. Convert back to BGR (3-channel) — required by PaddleOCR and NIM vision

    Returns a 3-channel uint8 BGR ndarray of the same spatial dimensions.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    # If the median pixel is dark, the background is dark — invert so text is dark on light
    if np.median(enhanced) < 127:
        enhanced = cv2.bitwise_not(enhanced)
        logger.debug("preprocess: dark background detected, image inverted")
    # Convert back to 3-channel BGR so PaddleOCR and vision models work correctly
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
