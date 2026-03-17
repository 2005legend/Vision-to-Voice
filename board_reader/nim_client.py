"""NIM VLM client for IntelliAgent Board Reader."""

from __future__ import annotations

import base64
import json
import logging
import time

import cv2
import numpy as np
import requests

from board_reader.config import Config

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    'You are a board content extractor. '
    'You must respond with ONLY a JSON object — no explanation, no markdown, no prose. '
    'JSON schema: {"topic": "<string>", "board_steps": [{"id": <int>, "text": "<string>"}], "equations": ["<string>"]}'
)

_USER_SUFFIX = (
    "\n\nRespond with ONLY the JSON object. No explanation. No markdown. Start your response with { and end with }."
)

_MODEL = "meta/llama-3.2-11b-vision-instruct"


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences and extract the JSON object from model output."""
    import re
    text = text.strip()
    # Remove opening/closing code fences
    if text.startswith("```"):
        text = text[text.find("\n") + 1:]
    if text.endswith("```"):
        text = text[:text.rfind("```")].rstrip()
    text = text.strip()
    # If model still wrapped JSON in prose, extract the first {...} block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def _parse_streamed_response(response: requests.Response) -> str:
    """Accumulate content from an SSE streamed NIM response."""
    parts: list[str] = []
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if line.startswith("data: "):
            line = line[6:]
        if line.strip() in ("[DONE]", ""):
            continue
        try:
            chunk = json.loads(line)
            delta = chunk["choices"][0].get("delta", {})
            parts.append(delta.get("content", ""))
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
    return "".join(parts)


def _encode_image(image: np.ndarray, max_side: int = 1280) -> str:
    """Encode a numpy image array to a base64 PNG string, resizing if needed."""
    h, w = image.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("cv2.imencode failed to encode image")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _build_payload(b64_image: str, ocr_text: str) -> dict:
    """Build the OpenAI-style multimodal request payload for a single image."""
    return _build_multi_payload([b64_image], ocr_text)


def _build_multi_payload(b64_images: list[str], ocr_text: str) -> dict:
    """Build payload for one or more images sent as a single VLM request."""
    user_content: list[dict] = [
        {"type": "text", "text": (
            f"These are {len(b64_images)} page(s) of a board/worksheet in order.\n"
            f"OCR text (all pages combined):\n{ocr_text}"
            + _USER_SUFFIX
        )}
    ]
    for b64 in b64_images:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    return {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 1024,
        "temperature": 0.1,
        "stream": False,
    }


def _do_request(payload: dict, config: Config) -> requests.Response:
    """Send the POST request to the NIM endpoint."""
    headers = {
        "Authorization": f"Bearer {config.nim_api_key}",
        "Content-Type": "application/json",
    }
    return requests.post(config.nim_endpoint, json=payload, headers=headers, timeout=30)


def _extract_json_via_groq(prose: str, config: Config) -> dict | None:
    """Use Groq to extract structured JSON from a prose VLM response."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return None

    if not config.groq_api_key:
        logger.error("NIM client: Groq key not set, cannot extract JSON from prose")
        return None

    schema = '{"topic": "<string>", "board_steps": [{"id": <int>, "text": "<string>"}], "equations": ["<string>"]}'
    prompt = (
        f"Extract the board content from this text and return ONLY a JSON object matching this schema:\n{schema}\n\n"
        f"Text to extract from:\n{prose[:2000]}\n\n"
        "Return ONLY the JSON object, nothing else."
    )

    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=config.groq_api_key)
        completion = client.chat.completions.create(
            model=config.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        text = completion.choices[0].message.content or ""
        text = _strip_code_fences(text)
        result = json.loads(text)
        logger.info("NIM client: Groq JSON extraction succeeded")
        return result
    except Exception as exc:
        logger.error("NIM client: Groq JSON extraction failed: %s", exc)
        return None


_GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def _call_groq_vision(b64_images: list[str], ocr_text: str, config: Config) -> dict | None:
    """Fallback: send image(s) directly to Groq vision model for board extraction."""
    if not config.groq_api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return None

    schema = '{"topic": "<string>", "board_steps": [{"id": <int>, "text": "<string>"}], "equations": ["<string>"]}'
    user_content: list[dict] = [
        {"type": "text", "text": (
            f"You are a board content extractor. Look at the board image(s) carefully.\n"
            f"OCR hint text:\n{ocr_text[:1000]}\n\n"
            f"Return ONLY a JSON object matching this schema:\n{schema}\n"
            "No explanation. No markdown. Start with {{ and end with }}."
        )}
    ]
    # Groq vision supports one image per message — use the first image only
    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64_images[0]}"},
    })

    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=config.groq_api_key)
        completion = client.chat.completions.create(
            model=_GROQ_VISION_MODEL,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.1,
            max_tokens=1024,
        )
        text = completion.choices[0].message.content or ""
        logger.info("Groq vision raw response: %r", text[:200])
        text = _strip_code_fences(text)
        result = json.loads(text)
        logger.info("NIM client: Groq vision fallback succeeded")
        return result
    except json.JSONDecodeError:
        # Groq vision returned prose — try text extraction
        logger.warning("NIM client: Groq vision returned prose, extracting JSON via text model")
        return _extract_json_via_groq(text, config)
    except Exception as exc:
        logger.error("NIM client: Groq vision fallback failed: %s", exc)
        return None


def call_nim_api(image: np.ndarray, ocr_text: str, config: Config) -> dict | None:
    """Call the NIM VLM endpoint with a single image. See call_nim_api_multi for multi-page."""
    return call_nim_api_multi([image], ocr_text, config)


def call_nim_api_multi(images: list[np.ndarray], ocr_text: str, config: Config) -> dict | None:
    """Call the NIM VLM endpoint with one or more images (multi-page support).

    Falls back to Groq vision if NIM fails.
    Returns the parsed JSON dict on success, or None on failure.
    """
    try:
        b64_images = [_encode_image(img) for img in images]
    except Exception as exc:
        logger.error("NIM client: failed to encode image(s): %s", exc)
        return None

    payload = _build_multi_payload(b64_images, ocr_text)

    for attempt in range(2):
        try:
            response = _do_request(payload, config)
            response.raise_for_status()
        except (requests.exceptions.RequestException, Exception) as exc:
            logger.error("NIM client: request error (attempt %d): %s", attempt + 1, exc)
            if attempt == 0:
                logger.info("NIM client: retrying after %.1f seconds", config.nim_retry_wait)
                time.sleep(config.nim_retry_wait)
                continue
            logger.warning("NIM client: all attempts failed, trying Groq vision fallback")
            return _call_groq_vision(b64_images, ocr_text, config)

        try:
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                content = _parse_streamed_response(response)
            else:
                data = response.json()
                content = data["choices"][0]["message"]["content"]

            if not content:
                logger.warning("NIM client: empty content from model, trying Groq vision fallback")
                return _call_groq_vision(b64_images, ocr_text, config)

            content_stripped = _strip_code_fences(content)
            try:
                return json.loads(content_stripped)
            except json.JSONDecodeError:
                logger.warning("NIM client: model returned prose, asking Groq to extract JSON")
                result = _extract_json_via_groq(content, config)
                if result is None:
                    logger.warning("NIM client: Groq text extraction failed, trying Groq vision fallback")
                    return _call_groq_vision(b64_images, ocr_text, config)
                return result

        except (KeyError, IndexError, TypeError) as exc:
            logger.error("NIM client: malformed response structure: %s", exc)
            logger.debug("NIM client: raw response body: %s", response.text[:500])
            logger.warning("NIM client: malformed response, trying Groq vision fallback")
            return _call_groq_vision(b64_images, ocr_text, config)

    return None
