# Vision-to-Voice: AI Classroom Reader

> **Accessibility-first smart assistant** — converts whiteboard content into real-time spoken explanations for visually impaired students in live classrooms.

---

## Overview

**Vision-to-Voice** is an AI-powered assistive technology designed for visually impaired students in live classroom settings. When a teacher writes on a whiteboard, this system:

1. Captures the whiteboard content (via webcam or uploaded image)
2. Extracts text and mathematical equations using OCR
3. Parses board structure using Vision Language Models (VLM)
4. Generates student-friendly spoken explanations via LLM
5. Speaks the explanation aloud using TTS
6. Allows the student to ask follow-up questions via STT
7. Adapts its teaching style over time using reinforcement learning

### Core Features

- **Real-time whiteboard monitoring** with change detection (skips redundant frames)
- **Multi-modal content understanding**: plain text, LaTeX equations, diagrams
- **Four explanation modes**: math, diagram, general, study (Socratic)
- **Push-to-talk STT** for hands-free follow-up questions
- **Doubt handler** for contextual Q&A
- **RL-based adaptation** that learns student's preferred explanation depth

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT SOURCES                               │
│  ┌─────────────────────┐     ┌─────────────────────┐            │
│  │  Mode 1: Webcam    │     │  Mode 2: Upload    │            │
│  │  (Live Feed)       │     │  (Manual Image)   │            │
│  └─────────┬──────────┘     └─────────┬──────────┘            │
│            │                           │                         │
│            ▼                         ▼                         │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │           capture.py                                  │     │
│  │  • OpenCV frame grab                                │     │
│  │  • Preprocessing (denoise, CLAHE, auto-invert)     │     │
│  │  • Change detection (pixel diff gate)                │     │
│  └─────────────────────┬───────────────────────────────┘     │
│                       │                                      │
│                       ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │              ocr.py                                   │     │
│  │  • PaddleOCR 2.6.2 (plain text extraction)          │     │
│  │  • pix2tex (LaTeX/mathequsation OCR)                  │     │
│  └─────────────────────┬───────────────────────────────┘     │
│                       │                                      │
│                       ▼                                      │
│  ┌───────────────────────────────��─────────────────────────┐     │
│  │           nim_client.py                               │     │
│  │  • NIM VLM API (NVIDIA AI Enterprise)                 │     │
│  │  • Groq Vision fallback                               │     │
│  │  • Returns structured JSON (topic, steps, equations)  │     │
│  └─────────────────────┬───────────────────────────────┘     │
│                       │                                      │
│                       ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │         intelliagent.py                                │     │
│  │  • Master orchestrator                                │     │
│  │  • Routes to correct explanation mode                 │     │
│  │  • Handles multi-call NIM                            │     │
│  │  • Applies RL adaptation                              │     │
│  └─────────────────────┬───────────────────────────────┘     │
│                       │                                      │
│            ┌──────────┴──────────┐                         │
│            ▼                      ▼                         │
│  ┌─────────────────┐    ┌─────────────────┐               │
│  │  gemini_client   │    │    doubt_handler │               │
│  │  (Explanation)  │    │  (Follow-up Q&A) │               │
│  └────────┬────────┘    └────────┬────────┘               │
│           │                       │                          │
│           ▼                       ▼                          │
│  ┌─────────────────┐    ┌─────────────────┐               │
│  │  tts_engine.py │    │  stt_engine.py │               │
│  │  (Windows SAPI)│    │  (faster-     │               │
│  │  TTS Audio    │    │   whisper)     │               │
│  └──────────────┘    └───────┬────────┘               │
│                              │                          │
│                              ▼                          │
│  ┌─────────────────────────────────────────────────┐         │
│  │              rl.py                              │         │
│  │  • Behaviour tracker                           │         │
│  │  • StudentProfile adaptation                   │         │
│  │  • Persists learning across sessions          │         │
│  └─────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology | Notes |
|-------|------------|-------|
| Frontend | Streamlit | Web UI for camera feed & controls |
| Frame Capture | OpenCV | Camera access, preprocessing |
| OCR | PaddleOCR 2.6.2 | **Pinned** — do not upgrade |
| LaTeX OCR | pix2tex | Equation extraction |
| VLM | NIM VLM API (LLaVA) | Primary board parser |
| VLM Fallback | Groq Vision | llama-4-scout-17b-16e-instruct |
| LLM | Groq | llama-3.3-70b-versatile |
| TTS | Windows SAPI | PowerShell subprocess |
| STT | faster-whisper | medium model |
| Config | YAML | config.yaml |
| Logging | Rotating file handler | logger.py |

---

## File Structure

```
PT2/
├── app.py                      # Streamlit UI entry point
├── board_reader/
│   ├── __init__.py
│   ├── __main__.py            # CLI entry point
│   ├── capture.py             # Frame capture, preprocessing, change detection
│   ├── ocr.py               # PaddleOCR + pix2tex
│   ├── nim_client.py          # NIM VLM API client + Groq fallback
│   ├── intelliagent.py       # Master orchestrator
│   ├── gemini_client.py      # Groq explanation engine (4 modes)
│   ├── tts_engine.py        # Windows SAPI TTS
│   ├── stt_engine.py       # faster-whisper STT
│   ├── doubt_handler.py     # Follow-up Q&A
│   ├── rl.py               # Reinforcement learning tracker
│   ├── models.py           # Dataclasses (BoardState, StudentProfile, etc.)
│   ├── session.py          # Session lifecycle manager
│   ├── config.py          # Config loader
│   ├── logger.py         # Rotating file logger
│   └── cli.py            # CLI interface
├── tests/
│   ├── unit/
│   └── property/
├── config.yaml.example
└── README.md
```

---

## Data Models

### BoardState
Represents the current whiteboard content:

```python
@dataclass
class BoardState:
    topic: str                          # Topic name (e.g., "Quadratic Equations")
    board_steps: List[BoardStep]         # Ordered steps on board
    equations: List[str]                # LaTeX formulas
```

### StudentProfile
Tracks student learning behavior for RL adaptation:

```python
@dataclass
class StudentProfile:
    grade_level: int                     # 10 or 12
    skip_count: int = 0                  # Times student skipped content
    replay_count: int = 0                 # Times student asked to repeat
    interrupt_count: int = 0               # Times student interrupted TTS
    repeat_count: int = 0                   # Repeat requests
    followup_count: int = 0               # Follow-up questions asked
    explanation_count: int = 0            # Total explanations given
    silence_duration: float = 0.0        # Seconds of silence
    preferred_detail: str = "medium"       # "brief" | "medium" | "detailed" | "slow"
```

### ChangeDelta
Represents what changed between two board states:

```python
@dataclass
class ChangeDelta:
    added_steps: List[BoardStep]          # New steps added
    changed_topic: str | None            # New topic (if changed)
    added_equations: List[str]            # New equations
```

---

## Pipeline Flow

### Input Modes

**Mode 1 — Webcam (Live Feed)**
1. OpenCV captures frames at configured interval
2. Change detection gate compares to previous frame
3. If pixel diff ≥ threshold → process
4. Otherwise → skip (saves API calls)

**Mode 2 — Upload Mode**
1. User uploads image via Streamlit UI
2. Bypasses capture.py → goes straight to OCR
3. Used for testing & demos

### Core Pipeline (Both Modes)

```
1. capture.py gets frame
2. change detection gate (webcam mode only)
3. ocr.py extracts text + LaTeX → BoardState
4. nim_client.py sends to VLM → structured JSON
5. intelliagent.py reads StudentProfile → selects mode
6. gemini_client.py generates explanation
7. tts_engine.py speaks via Windows SAPI
8. stt_engine.py listens for follow-up (push-to-talk)
9. If question → doubt_handler.py answers
10. Student signals feedback (understood/confused/skip)
11. rl.py updates StudentProfile
12. intelliagent.py uses updated profile on next cycle
```

---

## Configuration

### config.yaml (Example)

```yaml
# Student Configuration
student:
  grade_level: 10

# Camera Configuration
camera:
  index: 0                  # Camera device index
  capture_interval: 2.0     # Seconds between captures
  change_threshold: 0.15    # Pixel diff threshold (0.0–1.0)

# NIM VLM Configuration
nim:
  api_key: "YOUR_NIM_API_KEY"
  endpoint: "https://integrate.api.nvidia.com/v1/chat/completions"
  explain_model: "mistralai/mistral-small-24b-instruct"
  retry_wait: 3.0

# Groq Configuration (Primary LLM)
groq:
  api_key: "YOUR_GROQ_API_KEY"
  model: "llama-3.3-70b-versatile"

# TTS Configuration
tts:
  model: "sapi"           # "sapi" for Windows SAPI

# STT Configuration
stt:
  model: "medium"         # faster-whisper model size

# RL Configuration
rl:
  enabled: true
  profile_path: "student_profile.json"

# Logging Configuration
logging:
  level: "INFO"
  log_file: "app.log"
```

---

## Explanation Modes

### Math Mode (Default)
- Explains each step of a mathematical solution
- Reads equations in spoken words (e.g., "x equals negative four...")
- Explains WHAT and WHY for each step

### Diagram Mode
- For charts, biology figures, science diagrams
- Names and explains each labeled part
- Describes relationships between parts

### General Mode
- Describes any board content
- Useful for notes, instructions, schedules
- Covers all content, doesn't skip

### Study Mode (Socratic)
- First explains the topic
- Then asks 2-3 questions to test understanding
- Does not give away answers

---

## RL Adaptation Logic

The system adapts explanation style based on student behavior:

| Behavior Observed | Adaptation |
|-------------------|------------|
| >3 interrupts | Switch to "slow" (shorter sentences, pauses) |
| >2 repeat requests | Switch to "brief" (more concise) |
| >3 follow-ups | Switch to "detailed" (more thorough) |

Adaptation only fires after at least 5 explanations to avoid noise.

---

## Running the Application

### Streamlit UI (Recommended)

```bash
streamlit run app.py
```

### CLI Mode

```bash
python -m board_reader
```

### Debug Scripts

```bash
python debug_nim.py      # Test NIM VLM connection
python debug_ocr.py     # Test PaddleOCR
python debug_tts.py     # Test TTS output
python test_stt_standalone.py  # Test STT + VAD
```

---

## Testing

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test Suites

```bash
pytest tests/unit/           # Unit tests
pytest tests/property/        # Property-based tests
pytest tests/unit/test_intelliagent.py -v  # Specific file
```

### With Coverage

```bash
pytest tests/ --cov=board_reader --cov-report=term-missing
```

---

## Dependencies

```
opencv-python>=4.8.0
numpy>=1.24.0
paddleocr==2.6.2        # Pinned - DO NOT UPGRADE
pix2tex
requests
openai
pyyaml
sounddevice
faster-whisper
pyttsx3                  # TTS fallback
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Known Limitations

- **PaddleOCR**: Version 2.6.2 is pinned — upgrading breaks on Windows
- **Single-user**: Currently supports only one student profile
- **In-memory profile**: StudentProfile resets on app restart (no persistence)
- **TTS bleed**: STT sometimes fires on TTS audio leakage
- **Windows-only**: TTS uses Windows SAPI; macOS/Linux require different implementation

---

## Development Guidelines

### DO NOT
- Upgrade PaddleOCR above 2.6.2
- Use asyncio for TTS/STT (causes deadlocks)
- Pass raw PII to external APIs
- Block the main thread with TTS
- Swap Windows SAPI for Coqui without discussion

### ALWAYS
- Use `threading.Event` for push-to-talk
- Route LLM calls through `intelliagent.py`
- Check `BoardState.content_type` before choosing mode
- Log via `logger.py` — never bare `print()`
- Run debug scripts before modifying core modules

---

## License

MIT License — Sidaarth

---

## Support

- Issue Tracker: https://github.com/anomalyco/opencode/issues
- Project Owner: Sidaarth