"""OCR pipeline for IntelliAgent Board Reader.

Provides plain-text extraction (PaddleOCR 2.x), LaTeX formula extraction (pix2tex),
and a combiner that merges both into a single structured string.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_paddle_ocr_instance = None
_latex_ocr_instance = None


def _get_paddle_ocr():
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        from paddleocr import PaddleOCR  # type: ignore
        # PaddleOCR 2.x v2 API — use_angle_cls=True handles rotated text
        _paddle_ocr_instance = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            show_log=False,
        )
    return _paddle_ocr_instance


def _get_latex_ocr():
    global _latex_ocr_instance
    if _latex_ocr_instance is None:
        try:
            from pix2tex.cli import LatexOCR  # type: ignore
            _latex_ocr_instance = LatexOCR()
        except Exception as exc:
            logger.warning("LaTeX OCR (pix2tex) unavailable — torch/dll issue: %s. Skipping LaTeX OCR.", exc)
            _latex_ocr_instance = None  # stays None — extract_latex will return ""
    return _latex_ocr_instance


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text(image: np.ndarray) -> str:
    """Run PaddleOCR 2.x on *image* and return all detected text joined by newlines.

    PaddleOCR 2.x ocr() returns a list of pages. Each page is a list of:
      [[bbox_points], (text, confidence)]
    Returns ``""`` on any failure.
    """
    try:
        ocr = _get_paddle_ocr()
        logger.debug("PaddleOCR: running on image shape=%s dtype=%s", image.shape, image.dtype)

        results = ocr.ocr(image, cls=True)
        logger.debug("PaddleOCR: ocr returned %s pages", len(results) if results else 0)

        if not results:
            logger.warning("PaddleOCR: returned empty list")
            return ""

        lines: list[str] = []
        for page in results:
            if not page:
                continue
            for item in page:
                # item = [bbox, (text, confidence)]
                if item and len(item) >= 2:
                    text_conf = item[1]
                    if isinstance(text_conf, (list, tuple)) and text_conf:
                        t = str(text_conf[0]).strip()
                        conf = text_conf[1] if len(text_conf) > 1 else 1.0
                        if t:
                            lines.append(t)
                            logger.debug("PaddleOCR line: conf=%.2f text=%r", conf, t[:60])

        result_text = "\n".join(lines)
        logger.info("PaddleOCR: extracted %d lines, %d chars", len(lines), len(result_text))
        return result_text

    except Exception as exc:
        logger.error("PaddleOCR failed: %s", exc, exc_info=True)
        return ""


def extract_latex(image: np.ndarray) -> str:
    """Run LaTeX-OCR (pix2tex) on *image* and return the detected LaTeX string.

    Returns ``""`` silently if pix2tex is unavailable (torch/dll issues on Windows).
    """
    try:
        model = _get_latex_ocr()
        if model is None:
            return ""  # pix2tex unavailable — already warned at load time
        from PIL import Image as PILImage  # type: ignore
        pil_image = PILImage.fromarray(image)
        result = model(pil_image)
        if result is None:
            return ""
        logger.info("LaTeX OCR: result=%r", str(result)[:80])
        return str(result)
    except Exception as exc:
        logger.debug("LaTeX OCR failed: %s", exc)  # debug only — not an error
        return ""


def combine_ocr(text: str, latex: str) -> str:
    """Merge plain-text and LaTeX OCR results into a single structured string."""
    return f"[TEXT]\n{text}\n[LATEX]\n{latex}"
