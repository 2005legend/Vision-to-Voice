# CLAUDE.md — Vision-to-Voice (AI Classroom Reader)

## Project Identity

**Name:** Vision-to-Voice — Smart Classroom Assistant for the Visually Impaired  
**Type:** AI-powered assistive technology — converts whiteboard content into real-time spoken explanations for visually impaired students  
**Stage:** V2 active (V1 complete, V2 feedback loop + IntelliAgent layer in progress)  
**Owner:** Sidaarth  
**Platform:** Windows (primary), Python 3.10+  
**Entry Points:** `app.py` (Streamlit UI), `board_reader/__main__.py` (CLI)

> **Accessibility-first rule:** Every feature in this project exists to serve a visually impaired student in a live classroom. Never remove, simplify, or skip a pipeline stage because it seems redundant — each layer (OCR → VLM → explanation engine → TTS) is load-bearing for the core use case.

---

## What This Project Does

### Core Mission
A visually impaired student sits in a classroom. The teacher writes on the whiteboard. This system watches the board, understands what's written (text, math equations, diagrams), generates a student-friendly spoken explanation, and delivers it through audio — in real time, hands-free.

### V1 (Complete)
Static pipeline: user uploads a whiteboard/classroom image → OpenCV preprocessing → OCR extracts text and LaTeX → NIM VLM understands board context → Explanation engine generates student-friendly output → TTS speaks it aloud.

### V2 (Active Development)
Live feed pipeline with two input modes:

**Mode 1 — Software Mode (Laptop Webcam)**
- Continuous OpenCV frame capture with change detection (only processes frames with meaningful new content)
- Primary deployment mode

**Mode 2 — Upload Mode**
- User manually uploads an image
- Used for testing, demos, and debugging individual pipeline stages

Both modes feed into the same core pipeline:
- Real-time OCR + NIM VLM board parsing
- Groq LLM generates contextual explanation based on content type (math / diagram / general / study)
- TTS speaks explanation with pause/resume support
- **STT push-to-talk** lets student ask follow-up questions mid-explanation
- **Doubt handler** answers follow-ups using full board context
- **RL behaviour tracker** builds a StudentProfile and adapts explanation style over time
- **IntelliAgent layer** (`intelliagent.py`) orchestrates the full pipeline, self-corrects based on student feedback, and improves explanation quality across sessions

---

## Architecture Overview

```
Webcam/Image
     │
     ▼
capture.py          ← OpenCV frame grab + preprocess + change detection
     │
     ▼
ocr.py              ← PaddleOCR (text) + pix2tex (LaTeX)
     │
     ▼
nim_client.py       ← NIM VLM API for board parse (Groq vision fallback)
     │
     ▼
intelliagent.py     ← Pipeline orchestrator — routes to right explanation mode,
     │                 handles multi-call NIM, applies RL adaptation
     ▼
gemini_client.py    ← Groq explanation engine (math/diagram/general/study modes)
     │
     ├──► tts_engine.py    ← Windows SAPI TTS via PowerShell (pause/resume/interrupt)
     │
     └──► stt_engine.py    ← faster-whisper STT + dynamic VAD calibration
               │
               ▼
         doubt_handler.py  ← Groq follow-up Q&A with full board context injected
               │
               ▼
            rl.py          ← RL behaviour tracker → updates StudentProfile → feeds back
                             into intelliagent for next explanation cycle
```

---

## Tech Stack

| Layer | Technology | Decision Note |
|---|---|---|
| Frontend | Streamlit (`app.py`) | |
| Frame Capture | OpenCV | |
| OCR | PaddleOCR 2.6.2 (pinned) | **Do NOT upgrade** — breaks on Windows |
| LaTeX OCR | pix2tex | |
| VLM Board Parse | NIM VLM API (primary), Groq Vision (fallback) | NVIDIA NIM LLaVA-based |
| LLM Explanation | Groq — `llama-3.3-70b-versatile` | |
| TTS | Windows SAPI via PowerShell subprocess | **Chosen over Coqui/Mozilla TTS** — zero install friction on Windows, stable subprocess isolation. Do not swap back to Coqui without explicit instruction. |
| STT | faster-whisper (`medium` model), dynamic VAD | |
| Adaptation | Custom RL tracker (`rl.py`) + StudentProfile dataclass | |
| Orchestration | `intelliagent.py` | |
| Config | `config.yaml` (gitignored) — see `config.yaml.example` | |
| Logging | Rotating file logger (`logger.py`) | |
| Tests | pytest (unit) + Hypothesis (property-based) | |

---

## Key Files — What Each Does

| File | Role |
|---|---|
| `app.py` | Streamlit UI — camera feed, controls, session display |
| `board_reader/intelliagent.py` | **Master orchestrator** — `call_nim`, `call_nim_multi`, routes pipeline, applies RL corrections |
| `board_reader/capture.py` | Frame capture, preprocessing (denoise/threshold), change detection logic |
| `board_reader/ocr.py` | PaddleOCR text extraction + pix2tex LaTeX detection |
| `board_reader/nim_client.py` | NIM VLM API client + Groq vision fallback logic |
| `board_reader/gemini_client.py` | Groq explanation engine — 4 modes: math, diagram, general, study |
| `board_reader/tts_engine.py` | TTS via PowerShell SAPI — supports pause, resume, interrupt |
| `board_reader/stt_engine.py` | faster-whisper STT with push-to-talk via `threading.Event`, dynamic VAD |
| `board_reader/doubt_handler.py` | Follow-up Q&A — injects current board context into Groq prompt |
| `board_reader/rl.py` | RL behaviour tracker — logs student responses, updates StudentProfile |
| `board_reader/models.py` | Core dataclasses: `BoardState`, `BoardStep`, `StudentProfile` |
| `board_reader/session.py` | Session lifecycle — start, pause, resume, end |
| `board_reader/config.py` | Loads `config.yaml` → typed `Config` dataclass |
| `board_reader/logger.py` | Rotating file logger setup |
| `board_reader/cli.py` | CLI entry point |

---

## Critical Constraints — Read Before Touching Anything

### DO NOT
- **Upgrade PaddleOCR above 2.6.2** — breaks on Windows, this is pinned intentionally
- **Use asyncio for TTS/STT** — both use threading, mixing with asyncio causes deadlocks
- **Pass raw PII or student audio transcripts to external APIs** — route through doubt_handler which sanitizes context
- **Block the main thread with TTS** — TTS runs in a subprocess via PowerShell, keep it that way
- **Modify `config.yaml`** — it is gitignored; edit `config.yaml.example` for template changes
- **Swap Windows SAPI TTS for Coqui or Mozilla TTS** without explicit instruction — SAPI was chosen deliberately for zero-install Windows stability

### ALWAYS
- Use `threading.Event` for push-to-talk interrupt — do not replace with polling loops
- Route all LLM calls through `intelliagent.py` — never call `nim_client` or `gemini_client` directly from `app.py`
- Check `BoardState.content_type` before choosing explanation mode (math/diagram/general/study)
- Log all pipeline stages via `logger.py` — never use bare `print()` in production code
- Run `debug_*.py` scripts first when a specific module breaks before modifying core files

---

## Data Flow — V2 Full Pipeline

### Input routing (both modes feed step 3 onward)
```
Mode 1 (Webcam):  OpenCV → capture.py grabs frame → change detection gate
Mode 2 (Upload):  User uploads image → bypasses capture.py → goes straight to OCR
```

### Core pipeline (shared by all modes)
```
1.  capture.py / HTTP receiver gets frame
2.  Change detection — if insignificant change, discard and wait (webcam mode only)
3.  ocr.py extracts text (PaddleOCR) + LaTeX (pix2tex) → builds BoardState
4.  nim_client.py sends image + OCR text to NIM VLM → gets board context
5.  intelliagent.py receives BoardState + VLM output
6.  intelliagent reads StudentProfile (from rl.py) → selects explanation mode
    (math / diagram / general / study)
7.  gemini_client.py generates explanation via Groq
8.  tts_engine.py speaks explanation (PowerShell SAPI)
9.  stt_engine.py listens via push-to-talk (threading.Event) for student follow-up
10. If follow-up detected → doubt_handler.py answers using board context
11. Student signals feedback (understood / confused / skip)
12. rl.py updates StudentProfile
13. intelliagent uses updated profile on next cycle → adapts tone/depth/pacing
```

---

## StudentProfile & RL Adaptation

The `StudentProfile` dataclass (in `models.py`) tracks:
- Comprehension score per topic type (math, diagram, general)
- Preferred explanation depth (brief / detailed / step-by-step)
- Follow-up frequency (how often student asks doubts)
- Session history

`rl.py` updates this after every feedback signal. `intelliagent.py` reads the profile before generating each explanation prompt — this is the self-correction loop.

**When modifying rl.py or intelliagent.py:** always test the full feedback cycle, not just individual functions. The adaptation only reveals bugs across multiple explanation cycles.

---

## Development Commands

```bash
# Run Streamlit UI (V2 live feed)
streamlit run app.py

# Run CLI mode
python -m board_reader

# Debug individual modules (use these first when something breaks)
python debug_nim.py       # Test NIM VLM connection + fallback
python debug_ocr.py       # Test PaddleOCR on a sample image
python debug_tts.py       # Test TTS audio output
python test_stt_standalone.py  # Test STT + VAD in isolation

# Run all tests
pytest tests/

# Run only unit tests
pytest tests/unit/

# Run only property-based tests
pytest tests/property/

# Run a specific test file
pytest tests/unit/test_intelliagent.py -v

# Run with coverage
pytest tests/ --cov=board_reader --cov-report=term-missing
```

---

## Config Structure (config.yaml.example)

```yaml
groq:
  api_key: "YOUR_GROQ_API_KEY"
  model: "llama-3.3-70b-versatile"

nim:
  api_key: "YOUR_NIM_API_KEY"
  endpoint: "YOUR_NIM_ENDPOINT"

ocr:
  lang: "en"
  use_gpu: false          # Keep false on Windows unless CUDA confirmed

capture:
  camera_index: 0
  change_threshold: 0.15  # Frame diff threshold — tune per environment

tts:
  rate: 170               # Words per minute
  voice: "default"

stt:
  model: "medium"         # faster-whisper model size
  vad_threshold: 0.5

rl:
  learning_rate: 0.1
  decay: 0.95
```

---

## Test Coverage Map

| Module | Unit Test | Property Test |
|---|---|---|
| `capture.py` | `test_capture.py` | `test_prop_change.py`, `test_prop_preprocess.py` |
| `ocr.py` | `test_ocr.py` | `test_prop_ocr.py` |
| `nim_client.py` | `test_nim_client.py` | `test_prop_nim.py` |
| `intelliagent.py` | `test_intelliagent.py` | — |
| `gemini_client.py` | — | `test_prop_gemini.py` |
| `tts_engine.py` | `test_tts_engine.py` | `test_prop_tts.py` |
| `session.py` | `test_session.py` | — |
| `config.py` | `test_config.py` | `test_prop_config.py` |
| `rl.py` | — | `test_prop_rl.py` |
| `models.py` | — | `test_prop_board_state.py` |
| `cli.py` | `test_cli.py` | — |
| logger | — | `test_prop_logging.py` |

---

## Known Issues & Active Work

- **IntelliAgent multi-call (`call_nim_multi`):** Under active development — do not refactor until stable
- **STT VAD calibration:** Dynamic threshold works but sometimes fires on TTS audio bleed — issue tracked
- **RL adaptation depth:** Currently updates StudentProfile but `intelliagent.py` only uses comprehension score — full profile integration pending
- **Streamlit session state:** Some race conditions between live feed loop and Streamlit reruns — use `st.session_state` locks carefully

---

## V2 Pending Features (Do Not Implement Without Discussion)

- Multi-student profile support (currently single-user only)
- Persistent StudentProfile across sessions (currently in-memory only)
- Diagram understanding via NIM VLM — currently falls back to general mode
- Emotion/engagement detection from webcam (planned, not scoped)
- Auto-switch between input modes (webcam / upload) via config flag

---

## Claude Code Behaviour Rules for This Project

- **Never make changes to `ocr.py` PaddleOCR version or imports** without explicit instruction
- **Always edit `intelliagent.py` with extreme care** — it orchestrates the entire pipeline; a broken import here kills everything
- **When fixing bugs in `tts_engine.py` or `stt_engine.py`:** run the standalone debug scripts first, do not modify core until debug script confirms the fix
- **Prefer targeted single-function edits** over refactoring entire files
- **After any change to `models.py`:** check `rl.py`, `session.py`, and `intelliagent.py` for downstream breakage — all three depend on the dataclasses
- **Do not add new dependencies** without checking Windows compatibility first (especially audio libs)
- **Test the full feedback loop** (capture → explain → feedback → adapt → re-explain) before marking any V2 feature complete
