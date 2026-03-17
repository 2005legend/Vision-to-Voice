"""IntelliAgent Board Reader — Streamlit frontend."""

import logging
import os
import traceback

# Skip PaddleOCR/PaddleX connectivity check — avoids 10s delay on startup
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

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
from board_reader.rl import record_event, adapt_profile, load_profile, persist_profile

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IntelliAgent Board Reader",
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


def get_profile():
    """Load or initialise the StudentProfile for this session."""
    if "student_profile" not in st.session_state:
        cfg = get_config()
        st.session_state["student_profile"] = load_profile(cfg.rl_profile_path)
    return st.session_state["student_profile"]


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


# ── Pipeline helpers ──────────────────────────────────────────────────────────
def run_pipeline(frame_bgr: np.ndarray, mode: str = "math"):
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
            explanation = call_gemini_api(delta, board_state, config, mode=mode)
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


def run_multi_pipeline(frames_bgr: list[np.ndarray], mode: str = "math"):
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
            explanation = call_gemini_api(delta, board_state, config, mode=mode)
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
st.title("🎓 IntelliAgent Board Reader")
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
            if is_diagram_mode:
                # Diagram mode: explain first image only (NIM vision free-form)
                explanation = run_diagram_pipeline(frames_bgr[0])
                st.session_state["upload_explanation"] = explanation
                st.session_state["upload_board_state"] = None
                st.session_state["upload_is_diagram"] = True
            elif n == 1:
                board_state, explanation = run_pipeline(frames_bgr[0], explain_mode)
                st.session_state["upload_explanation"] = explanation
                st.session_state["upload_board_state"] = board_state
                st.session_state["upload_is_diagram"] = False
            else:
                board_state, explanation = run_multi_pipeline(frames_bgr, explain_mode)
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
            col_speak, col_doubt = st.columns([1, 1])
            with col_speak:
                if st.button("🔊 Speak explanation", key="speak_upload"):
                    speak(explanation)
                    st.success("Speaking…")
            with col_doubt:
                if st.button("🎤 Ask a doubt", key="doubt_upload"):
                    board_state_for_doubt = st.session_state.get("upload_board_state")
                    if board_state_for_doubt is None:
                        st.warning("No board context yet — run the pipeline first.")
                    else:
                        get_tts().pause()
                        with st.spinner("Listening… speak your question now"):
                            question = get_stt().listen(timeout=8.0)
                        get_tts().resume()
                        if question:
                            st.info(f"You asked: {question}")
                            profile = get_profile()
                            # Detect repeat request
                            if any(w in question.lower() for w in ["repeat", "say again", "again"]):
                                record_event(profile, "repeat")
                            else:
                                record_event(profile, "followup")
                            record_event(profile, "interrupt")
                            with st.spinner("Answering…"):
                                answer = handle_doubt(question, board_state_for_doubt, get_config(), profile)
                            adapt_profile(profile)
                            st.success(answer)
                            speak(answer)
                            st.session_state["upload_doubt_answer"] = answer
                        else:
                            st.warning("Didn't catch that — try again.")
                            get_tts().resume()

            # Show last doubt answer persistently
            if st.session_state.get("upload_doubt_answer"):
                with st.expander("💬 Last doubt answer", expanded=False):
                    st.write(st.session_state["upload_doubt_answer"])

# ── Tab 2: Camera ─────────────────────────────────────────────────────────────
with tab_camera:
    st.subheader("Live camera capture")

    if "cam_running" not in st.session_state:
        st.session_state.cam_running = False
    if "cam_frame" not in st.session_state:
        st.session_state.cam_frame = None

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🟢 Turn Camera ON", disabled=st.session_state.cam_running):
            st.session_state.cam_running = True
            st.rerun()
    with col_btn2:
        if st.button("🔴 Turn Camera OFF", disabled=not st.session_state.cam_running):
            st.session_state.cam_running = False
            st.session_state.cam_frame = None
            st.rerun()

    cam_placeholder = st.empty()

    if st.session_state.cam_running:
        cap = cv2.VideoCapture(cfg.camera_index)
        if not cap.isOpened():
            logging.getLogger("camera").error("Could not open camera index %d", cfg.camera_index)
            st.error(f"Could not open camera index {cfg.camera_index}. Check config.yaml.")
            st.session_state.cam_running = False
        else:
            st.info("Camera is ON — click Capture & Explain to process the current frame.")
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                st.session_state.cam_frame = frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cam_placeholder.image(rgb, caption="Live preview", use_container_width=True)

            if st.button("📸 Capture & Explain", type="primary", key="btn_cam"):
                if st.session_state.cam_frame is not None:
                    board_state, explanation = run_pipeline(st.session_state.cam_frame)
                    st.session_state["cam_explanation"] = explanation
                    st.session_state["cam_board_state"] = board_state
                    # Auto-speak as soon as explanation is ready
                    if explanation:
                        speak(explanation)

        # ── Persistent camera results ─────────────────────────────────────────
        cam_board = st.session_state.get("cam_board_state")
        cam_explanation = st.session_state.get("cam_explanation")

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
            col_speak_cam, col_doubt_cam = st.columns([1, 1])
            with col_speak_cam:
                if st.button("🔊 Speak explanation", key="speak_cam"):
                    speak(cam_explanation)
                    st.success("Speaking…")
            with col_doubt_cam:
                if st.button("🎤 Ask a doubt", key="doubt_cam"):
                    cam_board_for_doubt = st.session_state.get("cam_board_state")
                    if cam_board_for_doubt is None:
                        st.warning("No board context yet — capture a frame first.")
                    else:
                        get_tts().pause()
                        with st.spinner("Listening… speak your question now"):
                            question = get_stt().listen(timeout=8.0)
                        get_tts().resume()
                        if question:
                            st.info(f"You asked: {question}")
                            profile = get_profile()
                            if any(w in question.lower() for w in ["repeat", "say again", "again"]):
                                record_event(profile, "repeat")
                            else:
                                record_event(profile, "followup")
                            record_event(profile, "interrupt")
                            with st.spinner("Answering…"):
                                answer = handle_doubt(question, cam_board_for_doubt, get_config(), profile)
                            adapt_profile(profile)
                            st.success(answer)
                            speak(answer)
                        else:
                            st.warning("Didn't catch that — try again.")
                            get_tts().resume()
    else:
        cam_placeholder.info("Camera is OFF. Click Turn Camera ON to start.")
