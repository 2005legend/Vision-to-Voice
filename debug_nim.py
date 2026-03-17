"""Test NIM VLM endpoint directly and print the raw response."""
import numpy as np
import cv2
import base64
import json
import requests

api_key = "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"
endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"

# small test image — white text on black
img = np.zeros((100, 300, 3), dtype=np.uint8)
cv2.putText(img, "x = -2", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
_, buf = cv2.imencode(".png", img)
b64 = base64.b64encode(buf.tobytes()).decode()

system_prompt = (
    'You are a board parser. Return ONLY valid JSON matching this schema: '
    '{"topic": "<string>", "board_steps": [{"id": 1, "text": "<string>"}], "equations": ["<string>"]}'
)

payload = {
    "model": "meta/llama-3.2-11b-vision-instruct",
    "messages": [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "OCR text: x = -2"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        },
    ],
    "max_tokens": 256,
    "temperature": 0.1,
    "stream": False,
}

print("Sending request to NIM...")
try:
    r = requests.post(
        endpoint,
        json=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=30,
    )
    print(f"STATUS: {r.status_code}")
    print(f"CONTENT-TYPE: {r.headers.get('content-type', 'unknown')}")
    print(f"BODY:\n{r.text[:1000]}")
except Exception as e:
    print(f"REQUEST FAILED: {e}")
