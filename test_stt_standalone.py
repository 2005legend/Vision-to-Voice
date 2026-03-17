"""Standalone STT test — run this before wiring into the full pipeline.

Usage:
    python test_stt_standalone.py
"""
import sys
sys.path.insert(0, ".")

from board_reader.config import load_config
from board_reader.stt_engine import STTEngine

cfg = load_config("config.yaml")
print(f"Config stt.model = {cfg.stt_model}")

stt = STTEngine(cfg.stt_model)
print("Say something (up to 8 seconds, stops on silence)...")
result = stt.listen(timeout=8.0)
if result:
    print(f"Heard: {result}")
else:
    print("Nothing transcribed — check mic permissions or try a louder environment.")
