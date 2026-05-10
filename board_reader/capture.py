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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    if not cap.isOpened():
        logger.error("Camera index %d could not be opened.", camera_index)
        return None

    # Warm up camera
    for _ in range(3):
        cap.read()

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
      2. Denoise
      3. CLAHE (contrast enhancement)
      4. Auto-invert if background is dark (chalk board)
      5. Morphology ops (sharpening)
      6. Convert back to BGR (3-channel)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # CLAHE — adaptive contrast for board text
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Auto-invert if background is darker than text (chalk board)
    mean_val = cv2.mean(enhanced)[0]
    if mean_val < 120:  # dark background
        enhanced = 255 - enhanced
        logger.debug("preprocess: dark background detected, image inverted")

    # Slight sharpening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sharpened = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

    # Convert back to 3-channel BGR so PaddleOCR and vision models work correctly
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
