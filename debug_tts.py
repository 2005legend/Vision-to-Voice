"""Test TTS engine directly — speaks a test phrase twice to confirm repeated playback works."""
import time
from board_reader.tts_engine import TTSEngine, _speak_powershell

print("Test 1: direct _speak_powershell call")
_speak_powershell("Hello, this is the first test of the text to speech engine.")
print("Test 1 done")

time.sleep(1)

print("Test 2: second direct call (tests repeated playback)")
_speak_powershell("This is the second test. The quadratic formula gives x equals negative two.")
print("Test 2 done")

time.sleep(1)

print("Test 3: TTSEngine queue (tests the full pipeline)")
engine = TTSEngine()
engine.start()
engine.enqueue("Test three. The AI explanation is ready.")
time.sleep(0.5)
engine.enqueue("Step one: substitute the values into the formula.")
engine.stop(drain=True)
print("Test 3 done")
