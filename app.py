"""Vision To Voice — Streamlit frontend."""

import logging
import os
import traceback

# Skip PaddleOCR/PaddleX connectivity check — avoids 10s delay on startup
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
# Suppress OpenCV MSMF verbose frame-grab warnings on Windows
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from board_reader.config import load_config
from board_reader.capture import preprocess
from board_reader.ocr import combine_ocr, extract_text, extract_latex
from board_reader.intelliagent import call_nim, call_nim_multi, detect_change
from board_reader.gemini_client import call_gemini_api, explain_diagram
from board_reader.tts_engine import TTSEngine
from board_reader.stt_engine import STTEngine
from board_reader.doubt_handler import handle_doubt
from board_reader.rl import record_event, adapt_profile, load_profile, persist_profile, save_profile_on_change
from board_reader.models import StudentProfile

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vision To Voice",
    page_icon="🎓",
    layout="wide",
)

# ── In-memory log handler ─────────────────────────────────────────────────────
class _StreamlitLogHandler(logging.Handler):
    """Captures log records into st.session_state['log_lines'].

    Only writes from the main Streamlit thread — silently drops records from
    background threads (e.g. tts-worker) to avoid the ScriptRunContext flood.
    """

    def emit(self, record: logging.LogRecord) -> None:
        import threading
        if threading.current_thread().name != "MainThread":
            return
        try:
            if "log_lines" not in st.session_state:
                st.session_state["log_lines"] = []
            msg = self.format(record)
            st.session_state["log_lines"].append((record.levelname, msg))
            if len(st.session_state["log_lines"]) > 200:
                st.session_state["log_lines"] = st.session_state["log_lines"][-200:]
        except Exception:
            pass  # never let logging crash the app


def _install_log_handler() -> None:
    if st.session_state.get("_log_handler_installed"):
        return
    handler = _StreamlitLogHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    handler.setLevel(logging.DEBUG)
    # Also add a plain stderr handler so background thread logs (tts-worker etc.)
    # are still visible in the terminal without triggering ScriptRunContext warnings.
    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    stderr_handler.setLevel(logging.DEBUG)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Avoid adding duplicate handlers on Streamlit reruns
    if not any(isinstance(h, _StreamlitLogHandler) for h in root.handlers):
        root.addHandler(handler)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               and not isinstance(h, _StreamlitLogHandler) for h in root.handlers):
        root.addHandler(stderr_handler)
    st.session_state["_log_handler_installed"] = True


_install_log_handler()

# ── Config / TTS ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_config():
    cfg = load_config("config.yaml")
    logging.getLogger("config").info(
        "Config loaded. nim_explain_model=%r grade=%s",
        cfg.nim_explain_model, cfg.grade_level,
    )
    return cfg


@st.cache_resource
def get_tts():
    engine = TTSEngine()  # pyttsx3 — no model name needed
    engine.start()
    logging.getLogger("tts").info("TTSEngine started (pyttsx3)")
    return engine


@st.cache_resource
def get_stt():
    cfg = get_config()
    engine = STTEngine(model_size=cfg.stt_model)
    logging.getLogger("stt").info("STTEngine ready (model=%s)", cfg.stt_model)
    return engine


@st.cache_resource
def get_cam_state():
    return {
        "cam_running":            False,
        "live_mode_active":       False,
        "latest_frame":           None,
        "ai_running":             False,
        "last_explanation":       "",
        "last_spoken_explanation": "",
        "voice_trigger":          False,
        "ask_question_trigger":   False,
        "manual_capture_trigger": False,
        "camera_on_trigger":      False,
        "barge_in_enabled":       True,
        "voice_active":           False,
        "board_state":            None,
        "processed_frames":       0,
        "frame_timestamp":        0.0,
        "selected_camera_index":  1,   # default 1 = iVCam; overridden by sidebar
    }


def get_profile():
    """Load or initialise the StudentProfile for this session."""
    if "student_profile" not in st.session_state:
        cfg = get_config()
        st.session_state["student_profile"] = load_profile(cfg.rl_profile_path)
    return st.session_state["student_profile"]


def save_profile(profile: StudentProfile) -> None:
    """Persist the student profile to disk after changes."""
    cfg = get_config()
    persist_profile(profile, cfg.rl_profile_path)


def _safe_cfg_attr(cfg, attr: str, default=""):
    """Return cfg.attr safely — guards against stale cached Config objects."""
    val = getattr(cfg, attr, None)
    if val is None:
        logging.getLogger("config").warning(
            "Config missing attribute %r — cache may be stale. Click 'Reload config'.", attr
        )
        return default
    return val


def speak(text: str) -> None:
    get_tts().enqueue(text)


def _load_bgr(uploaded) -> np.ndarray:
    pil_img = Image.open(uploaded).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def _handle_voice_interaction(board_state, profile, mode="math"):
    """Speak explanation while simultaneously listening for student speech.

    Headphone mode: STT runs concurrently with TTS. Student speaks → TTS cuts → answer.
    Speaker mode:   TTS finishes first, then STT listens once.
    Never called from the main Streamlit thread.
    """
    if not board_state:
        return

    import threading
    import time

    stt = get_stt()
    tts = get_tts()
    cfg = get_config()
    # Read barge_in from cam_state (thread-safe) not st.session_state
    cam_state = get_cam_state()
    is_headphones = cam_state.get("barge_in_enabled", True)

    if not is_headphones:
        # Wait for TTS to drain, then listen once
        deadline = time.time() + 90.0
        while time.time() < deadline:
            if tts._queue.empty():
                time.sleep(0.5)
                break
            time.sleep(0.3)
        question = stt.listen(timeout=8.0)
        if question:
            tts.interrupt()
            _dispatch_voice_command(question, board_state, profile, mode)
        return

    # Headphone mode: concurrent listen + speak
    speech_detected = threading.Event()
    question_holder = [None]

    def _listen_worker():
        time.sleep(0.3)  # let TTS start first
        q = stt.listen(timeout=30.0)
        if q:
            question_holder[0] = q
            speech_detected.set()

    listener = threading.Thread(target=_listen_worker, daemon=True, name="stt-concurrent")
    listener.start()

    while not speech_detected.is_set():
        if tts._queue.empty():
            time.sleep(0.5)
            speech_detected.wait(timeout=2.0)
            break
        time.sleep(0.1)

    question = question_holder[0]
    if question:
        tts.interrupt()
        _dispatch_voice_command(question, board_state, profile, mode)


def _dispatch_voice_command(text: str, board_state, profile, mode: str = "math"):
    """Route a spoken command to the right action.

    Commands (case-insensitive):
      "repeat" / "say again" / "again"  → replay last explanation
      "stop" / "quiet" / "shut up"      → stop TTS, stay ready
      "capture" / "next" / "new photo"  → trigger a new capture
      "start live" / "live mode"        → start live monitoring
      "stop live" / "stop monitoring"   → stop live monitoring
      "camera on" / "turn on camera"    → turn camera on
      "camera off" / "turn off camera"  → turn camera off
      anything else                     → treat as a doubt question
    """
    import logging
    log = logging.getLogger("voice.cmd")
    t = text.lower().strip()
    tts = get_tts()
    cfg = get_config()

    log.info("Voice command received: %r", t)

    # ── Help ──────────────────────────────────────────
    if any(w in t for w in ["help", "what can you do", "commands", "options"]):
        speak(
            "Here are the voice commands. "
            "Say 'camera on' to start the camera. "
            "Say 'capture' or 'read board' to scan the board. "
            "Say 'start live' to monitor automatically. "
            "Say 'stop live' to stop monitoring. "
            "Say 'repeat' to hear the last explanation again. "
            "Say 'stop' to stop speaking. "
            "Or just ask any question about the board."
        )
        return

    # ── Stop ──────────────────────────────────────────
    if any(w in t for w in ["stop", "quiet", "shut up", "silence", "enough"]):
        tts.interrupt()
        speak("Okay, I'll stop.")
        return

    # ── Repeat ────────────────────────────────────────
    if any(w in t for w in ["repeat", "say again", "again", "replay", "once more"]):
        record_event(profile, "repeat")
        last = get_cam_state().get("last_spoken_explanation", "")
        if last:
            speak(last)
            _handle_voice_interaction(board_state, profile, mode)
        else:
            speak("Nothing to repeat yet.")
        return

    # ── Capture new photo ─────────────────────────────
    if any(w in t for w in ["capture", "next", "new photo", "take photo", "scan", "read board"]):
        cam_state = get_cam_state()
        cam_state["manual_capture_trigger"] = True
        speak("Capturing now.")
        return

    # ── Live mode on ──────────────────────────────────
    if any(w in t for w in ["start live", "live mode", "start monitoring", "auto mode"]):
        cam_state = get_cam_state()
        cam_state["live_mode_active"] = True
        speak("Live mode started. I'll speak whenever the board changes.")
        return

    # ── Live mode off ─────────────────────────────────
    if any(w in t for w in ["stop live", "stop monitoring", "manual mode", "pause live"]):
        cam_state = get_cam_state()
        cam_state["live_mode_active"] = False
        speak("Live mode stopped.")
        return

    # ── Camera on ─────────────────────────────────────
    if any(w in t for w in ["camera on", "turn on camera", "start camera", "open camera"]):
        cam_state = get_cam_state()
        cam_state["camera_on_trigger"] = True
        speak("Turning camera on.")
        return

    # ── Camera off ────────────────────────────────────
    if any(w in t for w in ["camera off", "turn off camera", "stop camera", "close camera"]):
        cam_state = get_cam_state()
        cam_state["cam_running"] = False
        cam_state["live_mode_active"] = False
        tts.interrupt()
        speak("Camera off.")
        return

    # ── Doubt / question ──────────────────────────────
    record_event(profile, "followup")
    answer = handle_doubt(text, board_state, cfg, profile)
    adapt_profile(profile)
    save_profile(profile)
    # Store for potential repeat — use cam_state (thread-safe)
    get_cam_state()["last_spoken_explanation"] = answer
    speak(answer)
    # Keep listening after answering
    _handle_voice_interaction(board_state, profile, mode)


# ── Pipeline helpers ──────────────────────────────────────────────────────────
def run_pipeline(frame_bgr: np.ndarray, mode: str = "math", profile: StudentProfile | None = None):
    config = get_config()
    log = logging.getLogger("pipeline.board")
    with st.status("Running pipeline…", expanded=True) as status:
        try:
            st.write("🔲 Preprocessing…")
            preprocessed = preprocess(frame_bgr)
            log.info("Preprocessing done. shape=%s", preprocessed.shape)

            st.write("🔤 OCR…")
            try:
                plain = extract_text(preprocessed)
                latex = extract_latex(preprocessed)
                ocr_text = combine_ocr(plain, latex)
                log.info("OCR done. plain=%d chars, latex=%d chars", len(plain), len(latex))
            except Exception as exc:
                log.warning("OCR failed: %s", exc, exc_info=True)
                ocr_text = combine_ocr("", "")
                st.warning(f"OCR skipped: {exc}")

            st.write("🤖 NIM VLM…")
            board_state = call_nim(preprocessed, ocr_text, config)
            if board_state is None:
                log.error("NIM VLM returned None")
                status.update(label="NIM VLM failed — see logs", state="error")
                return None, None
            log.info("NIM VLM ok. topic=%r steps=%d eqs=%d",
                     board_state.topic, len(board_state.board_steps), len(board_state.equations))

            st.write(f"📋 Topic: {board_state.topic or 'unknown'}")
            st.write("✨ AI explanation…")
            delta = detect_change(board_state, None)
            explanation = call_gemini_api(delta, board_state, config, profile=profile, mode=mode)
            if explanation is None:
                log.error("AI explanation returned None")
                status.update(label="AI explain failed — see logs", state="error")
                return board_state, None
            log.info("AI explanation ok. length=%d chars", len(explanation))
            status.update(label="Done!", state="complete")
        except Exception as exc:
            log.error("Pipeline error: %s\n%s", exc, traceback.format_exc())
            status.update(label=f"Error: {exc}", state="error")
            return None, None
    return board_state, explanation


def _run_pipeline_bg(frame_bgr, mode: str = "math", profile=None):
    """Pipeline for background threads — no Streamlit UI calls."""
    import logging
    log = logging.getLogger("pipeline.bg")
    config = get_config()
    try:
        preprocessed = preprocess(frame_bgr)
        try:
            plain = extract_text(preprocessed)
            latex = extract_latex(preprocessed)
            ocr_text = combine_ocr(plain, latex)
        except Exception:
            ocr_text = combine_ocr("", "")
        board_state = call_nim(preprocessed, ocr_text, config)
        if board_state is None:
            return None, None
        delta = detect_change(board_state, None)
        explanation = call_gemini_api(delta, board_state, config, profile=profile, mode=mode)
        return board_state, explanation
    except Exception as exc:
        log.error("BG pipeline error: %s", exc, exc_info=True)
        return None, None


def run_diagram_pipeline(frame_bgr: np.ndarray) -> str | None:
    """Run diagram analysis pipeline — NIM vision free-form description."""
    config = get_config()
    log = logging.getLogger("pipeline.diagram")
    with st.status("Analysing diagram…", expanded=True) as status:
        try:
            st.write("🔲 Preprocessing…")
            preprocessed = preprocess(frame_bgr)
            log.info("Preprocessing done. shape=%s", preprocessed.shape)

            st.write("🔬 NIM vision diagram description…")
            explanation = explain_diagram(preprocessed, config)
            if explanation is None:
                log.error("Diagram explanation returned None")
                status.update(label="Diagram analysis failed — see logs", state="error")
                return None
            log.info("Diagram explanation ok. length=%d chars", len(explanation))
            status.update(label="Done!", state="complete")
        except Exception as exc:
            log.error("Diagram pipeline error: %s\n%s", exc, traceback.format_exc())
            status.update(label=f"Error: {exc}", state="error")
            return None
    return explanation


def run_multi_pipeline(frames_bgr: list[np.ndarray], mode: str = "math", profile: StudentProfile | None = None):
    """Run the full pipeline across multiple pages/images as one coherent problem."""
    config = get_config()
    log = logging.getLogger("pipeline.multi")
    with st.status(f"Running pipeline on {len(frames_bgr)} page(s)…", expanded=True) as status:
        try:
            st.write("🔲 Preprocessing all pages…")
            preprocessed_list = [preprocess(f) for f in frames_bgr]
            log.info("Preprocessing done. pages=%d", len(preprocessed_list))

            st.write("🔤 OCR on all pages…")
            all_plain = []
            all_latex = []
            for i, pp in enumerate(preprocessed_list):
                try:
                    plain = extract_text(pp)
                    latex = extract_latex(pp)
                    all_plain.append(f"[Page {i+1}]\n{plain}")
                    all_latex.append(f"[Page {i+1}]\n{latex}")
                    log.info("OCR page %d: %d chars plain", i+1, len(plain))
                except Exception as exc:
                    log.warning("OCR page %d failed: %s", i+1, exc)
                    all_plain.append(f"[Page {i+1}]\n(OCR failed)")
                    all_latex.append("")
            ocr_text = combine_ocr("\n".join(all_plain), "\n".join(all_latex))

            st.write(f"🤖 NIM VLM ({len(preprocessed_list)} images)…")
            board_state = call_nim_multi(preprocessed_list, ocr_text, config)
            if board_state is None:
                log.error("NIM VLM returned None for multi-page input")
                status.update(label="NIM VLM failed — see logs", state="error")
                return None, None
            log.info("NIM VLM ok. topic=%r steps=%d eqs=%d",
                     board_state.topic, len(board_state.board_steps), len(board_state.equations))

            st.write(f"📋 Topic: {board_state.topic or 'unknown'}")
            st.write("✨ AI explanation…")
            delta = detect_change(board_state, None)
            explanation = call_gemini_api(delta, board_state, config, profile=profile, mode=mode)
            if explanation is None:
                log.error("AI explanation returned None")
                status.update(label="AI explain failed — see logs", state="error")
                return board_state, None
            log.info("AI explanation ok. length=%d chars", len(explanation))
            status.update(label="Done!", state="complete")
        except Exception as exc:
            log.error("Multi-page pipeline error: %s\n%s", exc, traceback.format_exc())
            status.update(label=f"Error: {exc}", state="error")
            return None, None
    return board_state, explanation
st.title("🎓 Vision To Voice")
st.caption("AI-powered board assistant for visually impaired students")

cfg = get_config()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Config")
    st.write(f"Grade: **{cfg.grade_level}th**")
    st.write(f"AI model: `{_safe_cfg_attr(cfg, 'groq_model', 'llama-3.3-70b-versatile')}` (Groq)")
    st.write(f"Vision model: `meta/llama-3.2-11b-vision-instruct` (NIM)")
    if st.button("🔄 Reload config", key="reload_cfg"):
        get_config.clear()
        get_tts.clear()
        get_stt.clear()
        st.rerun()
    st.caption("Edit `config.yaml` to change.")
    st.divider()

    # ── Camera selector ───────────────────────────────────────────────────────
    st.subheader("📷 Camera")
    cam_index_options = list(range(5))
    selected_cam = st.selectbox(
        "Camera index",
        cam_index_options,
        index=cam_index_options.index(cfg.camera_index) if cfg.camera_index in cam_index_options else 0,
        format_func=lambda i: f"Index {i} {'(config default)' if i == cfg.camera_index else ''}",
        help="0 = laptop camera, 1 = iVCam (phone). Change if wrong camera is used.",
        key="cam_index_select",
    )
    # Store in cam_state so capture thread uses it
    get_cam_state()["selected_camera_index"] = selected_cam
    st.caption(f"Active: index **{selected_cam}**. iVCam is usually index 1.")
    st.divider()

    # ── Audio Settings ────────────────────────────────────────────────────────
    st.subheader("🎧 Audio Settings")
    barge_in_mode = st.radio(
        "Audio Setup (Voice Control)",
        ["Headphones (Interrupt AI anytime)", "Speakers (Wait for AI to finish)"],
        index=0,
        help="Mandatory voice control is active. If using Speakers, wait for the explanation to finish before asking a question."
    )
    st.session_state["barge_in_enabled"] = barge_in_mode.startswith("Headphones")
    # Also write to cam_state so background threads can read it safely
    get_cam_state()["barge_in_enabled"] = st.session_state["barge_in_enabled"]
    st.divider()

    # ── TTS diagnostic ────────────────────────────────────────────────────────
    st.subheader("🔊 TTS Diagnostic")
    tts_test_text = st.text_input("Test phrase", value="Hello, the TTS engine is working.", key="tts_test_text")
    if st.button("▶ Test TTS", key="test_tts"):
        log_tts = logging.getLogger("test.tts")
        try:
            log_tts.info("TTS diagnostic: enqueuing %r", tts_test_text[:60])
            speak(tts_test_text)
            st.success("✅ Enqueued — you should hear audio now. Check logs if silent.")
            log_tts.info("TTS diagnostic: enqueue done")
        except Exception as exc:
            log_tts.error("TTS diagnostic error: %s", exc, exc_info=True)
            st.error(f"TTS error: {exc}")
    st.divider()

    # ── Individual endpoint testers ───────────────────────────────────────────
    st.subheader("🔧 Endpoint Testers")
    st.caption("Upload an image below, then run each endpoint. Results stay visible after each run.")

    test_img_file = st.file_uploader(
        "Test image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="sidebar_img"
    )

    # Show preview of uploaded test image
    if test_img_file is not None:
        st.image(Image.open(test_img_file).convert("RGB"), caption="Test image", use_container_width=True)

    def _get_test_preprocessed():
        if test_img_file is None:
            st.warning("Upload a test image above first.")
            return None
        frame = _load_bgr(test_img_file)
        return preprocess(frame), frame

    # 1. OCR — PaddleOCR
    with st.expander("1️⃣ OCR — PaddleOCR", expanded=bool(st.session_state.get("ocr_result"))):
        if st.button("▶ Run OCR", key="test_ocr"):
            result = _get_test_preprocessed()
            if result is not None:
                preprocessed, _ = result
                log = logging.getLogger("test.ocr")
                try:
                    log.info("OCR test started. shape=%s", preprocessed.shape)
                    text = extract_text(preprocessed)
                    log.info("OCR test done. chars=%d", len(text))
                    st.session_state["ocr_result"] = text or "(empty)"
                    st.session_state["ocr_status"] = f"✅ {len(text)} chars extracted"
                except Exception as exc:
                    log.error("OCR test error: %s", exc, exc_info=True)
                    st.session_state["ocr_result"] = f"ERROR: {exc}"
                    st.session_state["ocr_status"] = None
        if st.session_state.get("ocr_result"):
            st.text_area("PaddleOCR output", st.session_state["ocr_result"], height=140, key="ocr_out")
            if st.session_state.get("ocr_status"):
                st.success(st.session_state["ocr_status"])

    # 2. LaTeX OCR — pix2tex
    with st.expander("2️⃣ LaTeX OCR — pix2tex", expanded=bool(st.session_state.get("latex_result"))):
        if st.button("▶ Run LaTeX OCR", key="test_latex"):
            result = _get_test_preprocessed()
            if result is not None:
                preprocessed, _ = result
                log = logging.getLogger("test.latex")
                try:
                    log.info("LaTeX OCR test started. shape=%s", preprocessed.shape)
                    latex = extract_latex(preprocessed)
                    log.info("LaTeX OCR test done. result=%r", (latex or "")[:80])
                    st.session_state["latex_result"] = latex or "(empty)"
                    st.session_state["latex_status"] = f"✅ {len(latex)} chars extracted"
                except Exception as exc:
                    log.error("LaTeX OCR test error: %s", exc, exc_info=True)
                    st.session_state["latex_result"] = f"ERROR: {exc}"
                    st.session_state["latex_status"] = None
        if st.session_state.get("latex_result"):
            st.text_area("pix2tex output", st.session_state["latex_result"], height=100, key="latex_out")
            if st.session_state.get("latex_status"):
                st.success(st.session_state["latex_status"])

    # 3. VLM — NIM structured board parse
    with st.expander("3️⃣ VLM — NIM board parser", expanded=bool(st.session_state.get("vlm_result"))):
        if st.button("▶ Run VLM", key="test_vlm"):
            result = _get_test_preprocessed()
            if result is not None:
                preprocessed, _ = result
                log = logging.getLogger("test.vlm")
                try:
                    log.info("VLM test started. shape=%s", preprocessed.shape)
                    board_state = call_nim(preprocessed, "", cfg)
                    if board_state is None:
                        log.error("VLM test: call_nim returned None")
                        st.session_state["vlm_result"] = None
                        st.session_state["vlm_error"] = "VLM returned None — check logs"
                    else:
                        log.info("VLM test ok. topic=%r steps=%d eqs=%d",
                                 board_state.topic, len(board_state.board_steps), len(board_state.equations))
                        st.session_state["vlm_result"] = {
                            "topic": board_state.topic,
                            "steps": [{"id": s.id, "text": s.text} for s in board_state.board_steps],
                            "equations": board_state.equations,
                        }
                        st.session_state["vlm_error"] = None
                except Exception as exc:
                    log.error("VLM test error: %s", exc, exc_info=True)
                    st.session_state["vlm_result"] = None
                    st.session_state["vlm_error"] = str(exc)
        if st.session_state.get("vlm_result"):
            st.json(st.session_state["vlm_result"])
            st.success("✅ VLM parsed successfully")
        elif st.session_state.get("vlm_error"):
            st.error(st.session_state["vlm_error"])

    # 4. AI Explain — Groq (fast) with NIM fallback
    with st.expander("4️⃣ AI Explain — Groq / Mistral", expanded=bool(st.session_state.get("ai_result"))):
        test_prompt = st.text_area(
            "Custom prompt (optional)",
            value="Explain the quadratic formula step by step for a 10th grade student.",
            height=80,
            key="ai_prompt",
        )
        if st.button("▶ Run AI Explain", key="test_ai"):
            log = logging.getLogger("test.ai")
            try:
                from openai import OpenAI
                use_groq = bool(cfg.groq_api_key)
                if use_groq:
                    base_url = "https://api.groq.com/openai/v1"
                    api_key = cfg.groq_api_key
                    model = cfg.groq_model
                else:
                    base_url = "https://integrate.api.nvidia.com/v1"
                    api_key = cfg.nim_api_key
                    model = cfg.nim_explain_model
                log.info("AI explain test. provider=%s model=%s", "groq" if use_groq else "nim", model)
                client = OpenAI(base_url=base_url, api_key=api_key)
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": test_prompt}],
                    temperature=0.2,
                    top_p=0.7,
                    max_tokens=512,
                    stream=True,
                )
                parts = [c.choices[0].delta.content for c in completion
                         if c.choices[0].delta.content]
                response_text = "".join(parts)
                log.info("AI explain test done. length=%d chars", len(response_text))
                st.session_state["ai_result"] = response_text
                st.session_state["ai_status"] = f"✅ {len(response_text)} chars — {'Groq' if use_groq else 'NIM'}"
            except Exception as exc:
                log.error("AI explain test error: %s", exc, exc_info=True)
                st.session_state["ai_result"] = f"ERROR: {exc}"
                st.session_state["ai_status"] = None
        if st.session_state.get("ai_result"):
            st.text_area("AI response", st.session_state["ai_result"], height=200, key="ai_out")
            if st.session_state.get("ai_status"):
                st.success(st.session_state["ai_status"])

    st.divider()

    # ── Live log viewer ───────────────────────────────────────────────────────
    st.subheader("📋 Logs")
    if st.button("🗑 Clear logs", key="clear_logs"):
        st.session_state["log_lines"] = []

    log_lines = st.session_state.get("log_lines", [])
    if log_lines:
        # Color-code by level
        level_colors = {"ERROR": "🔴", "WARNING": "🟡", "INFO": "🟢", "DEBUG": "⚪"}
        log_text = "\n".join(
            f"{level_colors.get(lvl, '⚪')} {msg}" for lvl, msg in reversed(log_lines[-50:])
        )
        st.text_area("Recent logs (newest first)", log_text, height=300, key="log_viewer")
    else:
        st.caption("No logs yet — run a pipeline or endpoint test.")


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Camera"])

# ── Tab 1: Upload ─────────────────────────────────────────────────────────────
with tab_upload:
    st.subheader("Upload board photo(s)")

    mode = st.radio(
        "Mode",
        ["📐 Math", "🔬 Diagram / Biology", "📖 General", "🎓 Study"],
        horizontal=True,
        help=(
            "Math: OCR + NIM VLM + step-by-step explanation. "
            "Diagram: NIM vision free-form description. "
            "General: describe everything on the board. "
            "Study: explain then ask questions to test understanding."
        ),
    )
    is_diagram_mode = mode.startswith("🔬")
    explain_mode = (
        "math" if mode.startswith("📐")
        else "general" if mode.startswith("📖")
        else "study" if mode.startswith("🎓")
        else "diagram"
    )

    uploaded_files = st.file_uploader(
        "Choose image(s) — up to 5 pages",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="main_img",
    )

    if uploaded_files:
        # Cap at 5 pages
        if len(uploaded_files) > 5:
            st.warning("Maximum 5 pages supported. Using first 5.")
            uploaded_files = uploaded_files[:5]

        # Show thumbnails in a row
        cols = st.columns(len(uploaded_files))
        frames_bgr = []
        for i, uf in enumerate(uploaded_files):
            pil_img = Image.open(uf).convert("RGB")
            frames_bgr.append(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
            with cols[i]:
                st.image(pil_img, caption=f"Page {i+1}", use_container_width=True)

        n = len(frames_bgr)
        btn_label = (
            "🔬 Explain this diagram" if is_diagram_mode
            else f"▶ Explain {n} page{'s' if n > 1 else ''}"
        )

        if st.button(btn_label, key="btn_upload", type="primary"):
            profile = get_profile()
            if is_diagram_mode:
                # Diagram mode: explain first image only (NIM vision free-form)
                explanation = run_diagram_pipeline(frames_bgr[0])
                st.session_state["upload_explanation"] = explanation
                st.session_state["upload_board_state"] = None
                st.session_state["upload_is_diagram"] = True
            elif n == 1:
                board_state, explanation = run_pipeline(frames_bgr[0], explain_mode, profile)
                st.session_state["upload_explanation"] = explanation
                st.session_state["upload_board_state"] = board_state
                st.session_state["upload_is_diagram"] = False
            else:
                board_state, explanation = run_multi_pipeline(frames_bgr, explain_mode, profile)
                st.session_state["upload_explanation"] = explanation
                st.session_state["upload_board_state"] = board_state
                st.session_state["upload_is_diagram"] = False
            # Auto-speak as soon as explanation is ready
            if st.session_state.get("upload_explanation"):
                speak(st.session_state["upload_explanation"])

        # ── Persistent results (survive reruns) ───────────────────────────────
        board_state = st.session_state.get("upload_board_state")
        explanation = st.session_state.get("upload_explanation")
        is_diagram = st.session_state.get("upload_is_diagram", False)

        if board_state and not is_diagram:
            with st.expander("📋 Parsed board content", expanded=False):
                st.json({
                    "topic": board_state.topic,
                    "steps": [{"id": s.id, "text": s.text} for s in board_state.board_steps],
                    "equations": board_state.equations,
                })

        if explanation:
            st.subheader("🗣 Explanation")
            st.write(explanation)
            if st.button("🔊 Repeat explanation", key="speak_upload"):
                speak(explanation)
                st.success("Speaking…")


# ── Tab 2: Camera ─────────────────────────────────────────────────────────────
with tab_camera:
    st.subheader("Live camera capture")

    cam_state = get_cam_state()
    from board_reader.capture import compute_frame_diff

    # Register shutdown hook — stops threads cleanly when Streamlit exits
    import atexit
    def _shutdown():
        cam_state["cam_running"] = False
        cam_state["live_mode_active"] = False
    atexit.register(_shutdown)

    # ── Thread 1: Pure capture ───────────────────────
    def _capture_thread(cam_state: dict, camera_index: int):
        import time
        import os

        # Suppress OpenCV MSMF verbose warnings to stderr
        os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")

        log = logging.getLogger("capture")

        # Use default backend — iVCam requires MSMF (default on Windows), not DSHOW
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            log.error("Cannot open camera index %d", camera_index)
            cam_state["cam_running"] = False
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Warmup — discard first few frames
        for _ in range(5):
            cap.read()

        consecutive_failures = 0
        MAX_FAILURES = 30  # stop thread after 30 consecutive bad reads (~1s)

        while cam_state["cam_running"] and not cam_state.get("_restart_capture"):
            ret, frame = cap.read()
            if ret and frame is not None:
                cam_state["latest_frame"] = frame
                cam_state["frame_timestamp"] = time.time()
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    log.error("Camera lost after %d consecutive failures — stopping", MAX_FAILURES)
                    cam_state["cam_running"] = False
                    break
                time.sleep(0.033)

        cam_state["_restart_capture"] = False  # reset flag
        cap.release()
        log.info("Capture thread exited (index=%d)", camera_index)

    def _is_sharp_enough(frame, threshold=100.0):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() > threshold

    def _get_fresh_frame(cam_state, settle_ms=500):
        import time
        deadline = time.time() + 2.0
        last_seen = cam_state.get("frame_timestamp", 0)
        while time.time() < deadline:
            current_ts = cam_state.get("frame_timestamp", 0)
            if current_ts > last_seen + (settle_ms / 1000.0):
                return cam_state.get("latest_frame")
            time.sleep(0.05)
        return cam_state.get("latest_frame")

    # ── Thread 2: AI pipeline ────────────────────────
    def _ai_thread(cam_state: dict, capture_interval: float):
        import time
        prev_frame = None
        while cam_state["cam_running"]:
            if not cam_state["live_mode_active"]:
                time.sleep(0.1)
                continue

            snapshot = _get_fresh_frame(cam_state, settle_ms=500)

            if snapshot is None:
                time.sleep(1)
                continue

            if prev_frame is not None:
                diff = compute_frame_diff(snapshot, prev_frame)
                if diff < cfg.change_threshold:
                    time.sleep(2)       # nothing changed
                    continue

            if not _is_sharp_enough(snapshot):
                time.sleep(1)
                continue  # skip blurry frame

            prev_frame = snapshot.copy()
            cam_state["ai_running"] = True

            try:
                profile = get_profile()
                preprocessed = preprocess(snapshot)
                plain = extract_text(preprocessed)
                latex = extract_latex(preprocessed)
                ocr_text = combine_ocr(plain, latex)
                
                board_state = call_nim(preprocessed, ocr_text, cfg)
                
                if board_state is not None:
                    previous_board = cam_state.get("board_state")
                    delta = detect_change(board_state, previous_board)
                    
                    if delta is not None:
                        explanation = call_gemini_api(delta, board_state, cfg, profile=profile, mode="math")
                        
                        if explanation:
                            cam_state["last_explanation"] = explanation
                            cam_state["board_state"] = board_state
                            cam_state["processed_frames"] += 1
                            record_event(profile, "explanation")
                            cam_state["voice_trigger"] = True
            except Exception as e:
                logging.getLogger("live").error("AI pipeline error: %s", e, exc_info=True)
            finally:
                cam_state["ai_running"] = False

            time.sleep(capture_interval)

    # ── Thread 3: Always-on voice loop ──────────────────────────────
    def _voice_thread(cam_state: dict):
        """Always-on voice loop with non-blocking state machine.

        Uses a short STT timeout (2s) so the loop checks triggers every 2s max.
        State: IDLE → SPEAKING → LISTENING → IDLE
        """
        import time
        import threading
        log = logging.getLogger("voice")

        # Announce readiness once
        time.sleep(2.0)
        speak("Vision to Voice is ready. Say camera on to start, or say help for commands.")

        # Persistent idle listener thread — runs separately so voice_trigger
        # can be processed immediately without waiting for STT to time out
        idle_q = [None]          # holds last heard idle command
        idle_listening = [False] # True while idle listener is active

        def _idle_listener():
            while cam_state["cam_running"]:
                # Only listen when voice thread is truly idle
                if (not cam_state.get("voice_active")
                        and not cam_state.get("voice_trigger")
                        and not cam_state.get("manual_capture_trigger")):
                    idle_listening[0] = True
                    try:
                        q = get_stt().listen(timeout=3.0)
                        if q:
                            idle_q[0] = q
                    except Exception:
                        pass
                    finally:
                        idle_listening[0] = False
                else:
                    time.sleep(0.2)

        idle_t = threading.Thread(target=_idle_listener, daemon=True, name="stt-idle")
        idle_t.start()

        while cam_state["cam_running"]:

            # ── Process idle voice command ──────────────────────
            if idle_q[0] and not cam_state.get("voice_active"):
                q = idle_q[0]
                idle_q[0] = None
                board_state = cam_state.get("board_state")
                profile = get_profile()
                _dispatch_voice_command(q, board_state, profile, "math")
                time.sleep(0.1)
                continue

            # ── Manual capture trigger ───────────────────────────
            if cam_state.get("manual_capture_trigger"):
                cam_state["manual_capture_trigger"] = False
                cam_state["voice_active"] = True
                try:
                    if cam_state.get("latest_frame") is not None:
                        profile = get_profile()
                        board_state, explanation = _run_pipeline_bg(
                            cam_state["latest_frame"], "math", profile
                        )
                        if explanation:
                            cam_state["last_explanation"] = explanation
                            cam_state["board_state"] = board_state
                            cam_state["last_spoken_explanation"] = explanation
                            record_event(profile, "explanation")
                            speak(explanation)
                            _handle_voice_interaction(board_state, profile, "math")
                        else:
                            speak("Sorry, I couldn't read the board.")
                    else:
                        speak("Camera is not ready yet.")
                except Exception as e:
                    log.error("Manual capture error: %s", e, exc_info=True)
                    speak("Sorry, capture failed.")
                finally:
                    cam_state["voice_active"] = False
                continue

            # ── Camera on trigger ────────────────────────────────
            if cam_state.get("camera_on_trigger"):
                cam_state["camera_on_trigger"] = False
                if not cam_state["cam_running"]:
                    cam_state["cam_running"] = True
                    idx = cam_state.get("selected_camera_index", cfg.camera_index)
                    ensure_threads_running(cam_state, idx)
                continue

            # ── AI pipeline voice trigger ────────────────────────
            if cam_state.get("voice_trigger"):
                cam_state["voice_trigger"] = False
                cam_state["voice_active"] = True
                try:
                    explanation = cam_state["last_explanation"]
                    cam_state["last_spoken_explanation"] = explanation
                    speak(explanation)
                    profile = get_profile()
                    _handle_voice_interaction(cam_state["board_state"], profile, "math")
                except Exception as e:
                    log.error("Voice trigger error: %s", e, exc_info=True)
                finally:
                    cam_state["voice_active"] = False
                continue

            time.sleep(0.05)  # tight loop — checks triggers every 50ms

    # ── Thread launcher ──────────────────────────────
    def ensure_threads_running(cam_state: dict, camera_index: int):
        import threading
        # Always use the sidebar-selected index
        camera_index = cam_state.get("selected_camera_index", camera_index)
        threads = cam_state.setdefault("_threads", [])
        alive = [t for t in threads if t.is_alive()]
        cam_state["_threads"] = alive
        alive_names = {t.name for t in alive}

        # If capture thread is running but on a different index, kill it and restart
        last_index = cam_state.get("_active_camera_index", None)
        if "capture" in alive_names and last_index != camera_index:
            # Signal capture thread to stop — it will exit on next loop
            cam_state["_restart_capture"] = True
            # Remove from alive so it gets restarted below
            alive = [t for t in alive if t.name != "capture"]
            cam_state["_threads"] = alive
            alive_names = {t.name for t in alive}

        cam_state["_active_camera_index"] = camera_index

        targets = [
            ("capture", _capture_thread, (cam_state, camera_index)),
            ("ai",      _ai_thread,      (cam_state, cfg.capture_interval)),
            ("voice",   _voice_thread,   (cam_state,)),
        ]
        for name, fn, args in targets:
            if name not in alive_names:
                t = threading.Thread(target=fn, args=args, name=name, daemon=True)
                t.start()
                cam_state["_threads"].append(t)

    col1, col2, col3 = st.columns(3)
    with col1:
        if not cam_state["cam_running"]:
            if st.button("🟢 Turn Camera ON"):
                cam_state["cam_running"] = True
                # Use sidebar-selected index, not config default
                idx = cam_state.get("selected_camera_index", cfg.camera_index)
                ensure_threads_running(cam_state, idx)
                st.rerun()
        else:
            if st.button("🔴 Turn Camera OFF"):
                cam_state["cam_running"] = False
                cam_state["live_mode_active"] = False
                get_tts().interrupt()
                st.rerun()

    with col2:
        if cam_state["cam_running"]:
            if not cam_state["live_mode_active"]:
                if st.button("▶ Start Live Mode", type="primary"):
                    cam_state["live_mode_active"] = True
                    st.rerun()
            else:
                if st.button("⏹ Stop Live Mode", type="secondary"):
                    cam_state["live_mode_active"] = False
                    st.rerun()
        else:
            st.info("Click 'Turn Camera ON' first.")

    with col3:
        if st.button("🔇 Stop Speaking", key="stop_tts_cam"):
            get_tts().interrupt()
            st.success("Stopped.")

    # Manual ask-question button — works anytime camera is on
    if cam_state["cam_running"] and cam_state.get("board_state"):
        if st.button("🎤 Ask a Question", key="ask_question_cam", type="secondary"):
            cam_state["ask_question_trigger"] = True

    # ── Status indicators ─────────────────────────────────────────────────────
    status_cols = st.columns(3)
    with status_cols[0]:
        st.caption("🟢 Camera live" if cam_state["cam_running"] else "⚫ Camera off")
    with status_cols[1]:
        st.caption("🔵 AI running..." if cam_state["ai_running"] else "⚪ AI idle")
    with status_cols[2]:
        st.caption("🎙️ Listening..." if cam_state["voice_active"] else "⚪ Voice idle")

    if cam_state["live_mode_active"]:
        st.success(f"🔴 LIVE - Monitoring camera (processed {cam_state['processed_frames']} changes)")

    cam_placeholder = st.empty()

    if cam_state["cam_running"]:
        import base64
        def frame_to_base64(frame):
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return base64.b64encode(buffer).decode('utf-8')

        @st.fragment(run_every=0.1)
        def _video_feed():
            # Trigger full UI update when a new board state is fully processed
            current_processed = cam_state.get("processed_frames", 0)
            if "last_processed" not in st.session_state:
                st.session_state.last_processed = current_processed
            
            if current_processed > st.session_state.last_processed:
                st.session_state.last_processed = current_processed
                st.rerun()

            frame = cam_state.get("latest_frame")
            if frame is not None:
                b64 = frame_to_base64(frame)
                if cam_state.get("ai_running"):
                    html = f'''
                    <div style="position:relative; width:100%; aspect-ratio:16/9; border-radius:8px; overflow:hidden;">
                        <img src="data:image/jpeg;base64,{b64}" style="width:100%; height:100%; object-fit:cover; opacity:0.6;">
                        <div style="position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); 
                                    background:rgba(0,0,0,0.8); color:white; padding:15px 25px; border-radius:10px; 
                                    font-size:20px; font-weight:bold; font-family:sans-serif; text-align:center;
                                    box-shadow: 0 4px 6px rgba(0,0,0,0.3); z-index:10;">
                            ⏳ AI Processing Board...
                        </div>
                    </div>
                    '''
                else:
                    html = f'''
                    <div style="position:relative; width:100%; aspect-ratio:16/9; border-radius:8px; overflow:hidden;">
                        <img src="data:image/jpeg;base64,{b64}" style="width:100%; height:100%; object-fit:cover;">
                    </div>
                    '''
                cam_placeholder.html(html)
            else:
                cam_placeholder.info("Camera is warming up...")
                
        _video_feed()
        
        if not cam_state["live_mode_active"]:
            if st.button("📸 Capture & Explain", type="primary", key="btn_cam"):
                if cam_state["latest_frame"] is not None:
                    profile = get_profile()
                    board_state, explanation = run_pipeline(cam_state["latest_frame"], "math", profile)
                    cam_state["last_explanation"] = explanation
                    cam_state["board_state"] = board_state
                    if explanation:
                        record_event(profile, "explanation")
                        cam_state["voice_trigger"] = True  # voice thread picks this up

        cam_board = cam_state.get("board_state")
        cam_explanation = cam_state.get("last_explanation")

        if cam_board:
            with st.expander("📋 Parsed board content", expanded=False):
                st.json({
                    "topic": cam_board.topic,
                    "steps": [{"id": s.id, "text": s.text} for s in cam_board.board_steps],
                    "equations": cam_board.equations,
                })

        if cam_explanation:
            st.subheader("🗣 Explanation")
            st.write(cam_explanation)
            col_speak_cam, col_feedback = st.columns([1, 1])
            with col_speak_cam:
                if st.button("🔊 Repeat explanation", key="speak_cam"):
                    speak(cam_explanation)
                    st.success("Speaking…")

            with col_feedback:
                st.caption("Was this helpful?")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("✅ Yes", key="feedback_yes"):
                        profile = get_profile()
                        record_event(profile, "explanation")
                        st.success("Thanks!")
                with col_no:
                    if st.button("❌ Confused", key="feedback_no"):
                        profile = get_profile()
                        record_event(profile, "interrupt")
                        adapt_profile(profile)
                        save_profile(profile)
                        st.warning("I'll simplify the next explanation.")
    else:
        cam_placeholder.info("Camera is OFF. Click Turn Camera ON to start.")
