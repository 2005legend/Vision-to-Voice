"""Run this standalone to see exactly what PaddleOCR 3.x returns."""
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
# Disable oneDNN (MKL-DNN) — crashes on Windows with Paddle 3.x (ConvertPirAttribute2RuntimeAttribute)
os.environ["FLAGS_use_mkldnn"] = "0"

import sys
import numpy as np
import cv2

# Use the image passed as argument, or create a synthetic one
if len(sys.argv) > 1:
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not load {sys.argv[1]}")
        sys.exit(1)
    print(f"Loaded image: {img.shape}")
else:
    # Synthetic white image with black text
    img = np.ones((200, 600, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Why Me?", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(img, "Hello World Test", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    print(f"Using synthetic image: {img.shape}")

from paddleocr import PaddleOCR

print("Initialising PaddleOCR...")
ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

print("Running ocr()...")
results = ocr.ocr(img, cls=True)

print(f"\n--- RAW RESULT ---")
print(f"Type: {type(results)}")
print(f"Length: {len(results) if results else 0}")

if results:
    for i, res in enumerate(results):
        print(f"\nresult[{i}] type: {type(res)}")
        if isinstance(res, dict):
            print(f"  keys: {list(res.keys())}")
            for k, v in res.items():
                print(f"  [{k}] = {repr(v)[:200]}")
        elif hasattr(res, "__dict__"):
            print(f"  attrs: {list(res.__dict__.keys())}")
        else:
            print(f"  value: {repr(res)[:300]}")
            try:
                for j, item in enumerate(res):
                    print(f"    item[{j}]: {repr(item)[:200]}")
            except Exception as e:
                print(f"    not iterable: {e}")
