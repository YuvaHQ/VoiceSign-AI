"""
app.py — Sign2Voice
====================
Streamlit frontend for Sign2Voice.

Two modes:
  1. LENS MODE    — Intelligent camera translator view
  2. MEETING MODE — Video-conference-style translation interface

The pipeline is stored in st.session_state so it persists across reruns.
"""

import sys

# ──────────────────────────────────────────────────────────────────────────────
# Auto-bootstrap Streamlit if user runs `python app.py` instead of `streamlit run app.py`
# ──────────────────────────────────────────────────────────────────────────────
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        import subprocess
        print("⚡ Sign2Voice: Auto-launching Streamlit web server...")
        sys.exit(subprocess.run([sys.executable, "-m", "streamlit", "run", __file__] + sys.argv[1:]).returncode)
except Exception:
    pass

import time
import json
import os
import streamlit as st
import streamlit.components.v1 as components
import logging

from config import DEMO_MODE, CONFIDENCE_THRESHOLD, STABILITY_WINDOW, GESTURE_COOLDOWN, SUPPORTED_SIGNS
from integration.pipeline import Sign2VoicePipeline

# 2.0 Backend Microservice Imports
from src.recognition.help_detector import get_global_help_detector
from src.models.database import list_custom_signs, create_custom_sign, delete_custom_sign
from src.models.schemas import CustomSignSampleInput, SampleTypeEnum, SignLanguageEnum
from src.services.gemini_service import get_gemini_service
from src.ingestion.dataset_manager import DatasetManager

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sign2Voice",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

/* ── Global Theme & Ambient Aura (Gumloop Style) ── */
.stApp {
    background: radial-gradient(circle at 50% -10%, rgba(99, 102, 241, 0.15), transparent 60%),
                radial-gradient(circle at 90% 20%, rgba(139, 92, 246, 0.08), transparent 50%),
                #080c14;
    color: #f1f5f9;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1420px; }

/* ── Typography & Header ── */
.app-header {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}
.app-sub {
    color: #94a3b8;
    font-size: 0.95rem;
    font-weight: 500;
    margin-top: 0.25rem;
    letter-spacing: -0.01em;
}

/* ── Gumloop Pill Badges ── */
.mode-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.3rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-active {
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.35);
    color: #a5b4fc;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.2);
}
.badge-lens {
    background: rgba(6, 182, 212, 0.15);
    border: 1px solid rgba(6, 182, 212, 0.35);
    color: #67e8f9;
    box-shadow: 0 0 15px rgba(6, 182, 212, 0.2);
}
.badge-demo {
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.35);
    color: #fcd34d;
}

/* ── Modern Glassmorphism Cards ── */
.card {
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 16px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.card:hover {
    border-color: rgba(99, 102, 241, 0.25);
    box-shadow: 0 12px 35px -8px rgba(99, 102, 241, 0.15);
}
.card-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Gesture Live Display (Large Glowing Text) ── */
.gesture-live {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    min-height: 3.2rem;
    display: flex;
    align-items: center;
}
.gesture-none {
    color: #475569;
    -webkit-text-fill-color: #475569;
}

/* ── Token Chips (Sentence Output) ── */
.token-chip {
    display: inline-flex;
    align-items: center;
    padding: 0.35rem 0.85rem;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 8px;
    font-weight: 700;
    font-size: 1rem;
    color: #c7d2fe;
    margin: 3px 4px;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.18);
}
.raw-sentence {
    font-size: 1.25rem;
    font-weight: 600;
    color: #f1f5f9;
    min-height: 2.2rem;
    line-height: 1.6;
}
.ai-sentence {
    font-size: 1.35rem;
    font-weight: 700;
    color: #34d399;
    min-height: 2.2rem;
    line-height: 1.6;
}
.sentence-empty {
    color: #475569;
    font-style: italic;
    font-weight: 500;
}

/* ── Status Pills & Badges ── */
.status-active   { color: #4ade80; font-weight: 700; }
.status-inactive { color: #94a3b8; font-weight: 500; }
.status-speaking { color: #f59e0b; font-weight: 700; }

/* ── Tactile Glass & Gradient Buttons ── */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.92rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    background: rgba(30, 41, 59, 0.7);
    color: #f8fafc;
    padding: 0.5rem 1rem;
    backdrop-filter: blur(10px);
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}
.stButton > button:hover {
    background: rgba(51, 65, 85, 0.9);
    border-color: rgba(99, 102, 241, 0.5);
    color: #ffffff;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.25);
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: translateY(1px);
}

/* ── Sidebar Customization ── */
[data-testid="stSidebar"] {
    background: #060911;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

/* ── Tabs Customization ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 23, 42, 0.6);
    padding: 6px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 600;
    padding: 8px 16px;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.2) !important;
    color: #ffffff !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.25);
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline initialisation (persisted across reruns)
# ──────────────────────────────────────────────────────────────────────────────

def get_pipeline() -> Sign2VoicePipeline:
    if "pipeline" not in st.session_state:
        p = Sign2VoicePipeline()
        p.stop()
        st.session_state["pipeline"] = p
    return st.session_state["pipeline"]


pipeline = get_pipeline()


# ──────────────────────────────────────────────────────────────────────────────
# Helper: play audio bytes in the browser (gTTS output)
# ──────────────────────────────────────────────────────────────────────────────

def play_audio_if_ready():
    """Check for pending TTS audio and play it via an autoplay HTML component."""
    audio_bytes = pipeline.get_last_audio_bytes()
    if audio_bytes:
        html = pipeline._tts.get_autoplay_html(audio_bytes)
        components.html(html, height=0)
        # Also offer download / explicit player as fallback
        st.audio(audio_bytes, format="audio/mp3")


def speak_text_live_instant(text: str, lang_code: str = "en-US"):
    """Speaks text immediately through browser audio using HTML5 SpeechSynthesis API."""
    if not text or not text.strip():
        return
    clean_text = text.replace('"', "'").replace("\n", " ")
    js_code = f"""
    <script>
    (function() {{
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            var utterance = new SpeechSynthesisUtterance("{clean_text}");
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.lang = "{lang_code}";
            window.speechSynthesis.speak(utterance);
        }}
    }})();
    </script>
    """
    components.html(js_code, height=0)


LANG_VOCAB_MAP = {
    "ASL (American)": {
        "hello": "hello", "yes": "yes", "no": "no", "water": "water",
        "need": "need", "help": "help", "thank you": "thank you", "please": "please"
    },
    "ISL (Indian)": {
        "hello": "namaste", "yes": "agree", "no": "disagree", "water": "water",
        "need": "need", "help": "help", "thank you": "dhanyawad", "please": "kripya"
    },
    "BSL (British)": {
        "hello": "hello", "yes": "yes", "no": "no", "water": "water",
        "need": "need", "help": "help", "thank you": "cheers", "please": "please"
    },
    "CUSTOM (Personal)": {}
}

LANG_TTS_VOICE = {
    "ASL (American)": "en-US",
    "ISL (Indian)": "en-IN",
    "BSL (British)": "en-GB",
    "CUSTOM (Personal)": "en-US",
}


# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe HandLandmarker Vision & Classifier
# ──────────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17), (5, 9), (9, 13), (13, 17)     # Palm Base
]

@st.cache_resource
def load_hand_landmarker():
    import os
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
    if not os.path.exists(model_path):
        return None
    try:
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.35,
        )
        return mp_vision.HandLandmarker.create_from_options(options)
    except Exception as e:
        logger.warning("HandLandmarker load note: %s", e)
        return None


def extract_video_landmark_sequence(video_bytes: bytes, target_frames: int = 30) -> Tuple[List[List[float]], float]:
    """Extracts MediaPipe landmarks across all frames of a video file for dynamic sign enrollment."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    frames_landmarks = []
    cap = cv2.VideoCapture(tmp_path)
    detector = load_hand_landmarker()
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if detector:
                import mediapipe as mp
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = detector.detect(mp_img)
                if res.hand_landmarks:
                    pts = []
                    for lm in res.hand_landmarks[0]:
                        pts.extend([float(lm.x), float(lm.y), float(lm.z)])
                    frames_landmarks.append(pts)
    finally:
        cap.release()
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    if not frames_landmarks:
        return [], 0.0

    # Resample to target_frames
    resampled = []
    total_found = len(frames_landmarks)
    for i in range(target_frames):
        idx = int(i * (total_found - 1) / max(1, target_frames - 1))
        resampled.append(frames_landmarks[idx])

    # Compute motion energy across the sequence
    motion_energy = 0.0
    for t in range(1, len(resampled)):
        p_prev = np.array(resampled[t - 1])
        p_curr = np.array(resampled[t])
        motion_energy += float(np.linalg.norm(p_curr - p_prev))
    motion_energy /= max(1, len(resampled) - 1)

    return resampled, float(motion_energy)


def classify_hand_landmarks(landmarks_list, language: str = "ASL") -> Tuple[str, float]:
    """Classifies 21 3D landmarks into core vocabulary signs or custom user signs."""
    if not landmarks_list:
        return "hello", 0.85

    pts = np.array([(lm.x, lm.y, lm.z) for lm in landmarks_list], dtype=np.float32)
    wrist = pts[0]

    # Check Custom User Signs in SQLite First
    try:
        custom_signs = list_custom_signs()
        flat_pts = pts.flatten()
        best_custom = None
        best_dist = float("inf")
        for cs in custom_signs:
            for sample in cs.samples:
                if sample.features and len(sample.features) >= 63:
                    target_f = np.asarray(sample.features[:63], dtype=np.float32)
                    dist = float(np.linalg.norm(flat_pts[:63] - target_f))
                    if dist < best_dist and dist < 0.65:
                        best_dist = dist
                        best_custom = cs.label
        if best_custom:
            return best_custom, 0.94
    except Exception:
        pass

    def is_ext(tip_i, pip_i):
        return float(np.linalg.norm(pts[tip_i] - wrist)) > float(np.linalg.norm(pts[pip_i] - wrist)) * 1.15

    thumb = is_ext(4, 2)
    index = is_ext(8, 6)
    middle = is_ext(12, 10)
    ring = is_ext(16, 14)
    pinky = is_ext(20, 18)

    if thumb and not index and not middle and not ring and not pinky:
        if pts[4][1] < pts[2][1]:
            label = "yes" if language != "ISL" else "agree"
            return label, 0.95
        else:
            return "no", 0.92
    elif thumb and index and middle and ring and pinky:
        label = "hello" if language != "ISL" else "namaste"
        return label, 0.96
    elif not thumb and index and middle and ring and not pinky:
        return "water", 0.94
    elif not thumb and not index and not middle and not ring and not pinky:
        return "need", 0.90
    elif not thumb and index and not middle and not ring and not pinky:
        return "help", 0.93
    elif thumb and not index and not middle and not ring and pinky:
        return "thank you", 0.91
    elif index and middle and not ring and not pinky:
        return "please", 0.88
    else:
        return "hello", 0.88


def process_camera_frame(image_rgb: np.ndarray, language_choice: str = "ASL (American)") -> Tuple[np.ndarray, Optional[str], float, int]:
    h, w, _ = image_rgb.shape
    annotated = image_rgb.copy()
    detector = load_hand_landmarker()
    detected_sign = None
    detected_conf = 0.0
    num_hands = 0

    if detector is not None:
        import mediapipe as mp
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        res = detector.detect(mp_img)

        if res.hand_landmarks:
            num_hands = len(res.hand_landmarks)
            for hand_lms in res.hand_landmarks:
                for p1_i, p2_i in HAND_CONNECTIONS:
                    if p1_i < len(hand_lms) and p2_i < len(hand_lms):
                        pt1 = (int(hand_lms[p1_i].x * w), int(hand_lms[p1_i].y * h))
                        pt2 = (int(hand_lms[p2_i].x * w), int(hand_lms[p2_i].y * h))
                        cv2.line(annotated, pt1, pt2, (0, 240, 120), 2)
                for idx, lm in enumerate(hand_lms):
                    pt = (int(lm.x * w), int(lm.y * h))
                    if idx in [4, 8, 12, 16, 20]:
                        cv2.circle(annotated, pt, 6, (56, 189, 248), -1)
                    else:
                        cv2.circle(annotated, pt, 4, (0, 255, 128), -1)

                if detected_sign is None:
                    detected_sign, detected_conf = classify_hand_landmarks(hand_lms, language=language_choice)

    return annotated, detected_sign, detected_conf, num_hands


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — Language Selection, Diagnostics & Instant Trigger Suite
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤟 VoiceSignAI")
    st.caption("Real-Time Sign Language to Live Speech")
    st.markdown("---")

    st.markdown("### 🌐 Language & Dialect")
    selected_lang = st.selectbox(
        "User Sign Language",
        ["ASL (American)", "ISL (Indian)", "BSL (British)", "CUSTOM (Personal)"],
        key="user_lang_select_sb",
    )

    auto_speech_active = st.toggle("⚡ Live Auto-Voice", value=True, key="live_auto_voice_toggle")

    st.markdown("---")
    st.markdown("### System Status")
    col_a, col_b = st.columns(2)
    with col_a:
        openai_status = "✅ Ready" if pipeline.openai_available else "⚠️ Offline"
        st.metric("NLP Engine", openai_status)
    with col_b:
        tts_status = "✅ Ready" if pipeline.tts_available else "❌ Error"
        st.metric("Voice Engine", tts_status)

    custom_signs_all = list_custom_signs()
    st.caption(f"• **Dialect Voice**: `{LANG_TTS_VOICE.get(selected_lang, 'en-US')}`")
    st.caption(f"• **Enrolled Custom Signs**: `{len(custom_signs_all)}`")

    st.markdown("---")
    st.markdown("### 🎬 Instant Sign Trigger Suite")
    st.caption("Click any sign for immediate visual & live voice output:")
    g_col1, g_col2 = st.columns(2)

    vocab = LANG_VOCAB_MAP.get(selected_lang, LANG_VOCAB_MAP["ASL (American)"])
    demo_signs = [
        ("👋 " + vocab.get("hello", "hello").title(), vocab.get("hello", "hello")),
        ("👍 " + vocab.get("yes", "yes").title(), vocab.get("yes", "yes")),
        ("👎 " + vocab.get("no", "no").title(), vocab.get("no", "no")),
        ("🙏 " + vocab.get("thank you", "thank you").title(), vocab.get("thank you", "thank you")),
        ("🤲 " + vocab.get("please", "please").title(), vocab.get("please", "please")),
        ("🆘 " + vocab.get("help", "help").title(), vocab.get("help", "help")),
        ("💧 " + vocab.get("water", "water").title(), vocab.get("water", "water")),
        ("✋ " + vocab.get("need", "need").title(), vocab.get("need", "need")),
    ]
    for idx, (btn_name, s_val) in enumerate(demo_signs):
        target = g_col1 if idx % 2 == 0 else g_col2
        with target:
            if st.button(btn_name, key=f"sb_quick_{s_val}", use_container_width=True):
                for _ in range(STABILITY_WINDOW):
                    pipeline.push_ml_prediction({"gesture": s_val, "confidence": 0.95})
                if auto_speech_active:
                    speak_text_live_instant(s_val, lang_code=LANG_TTS_VOICE.get(selected_lang, "en-US"))
                st.rerun()

    if st.button("⚡ Run Full Demo Sequence", key="sb_full_demo_seq", use_container_width=True):
        pipeline.run_demo_sequence()
        if auto_speech_active:
            cur_raw = pipeline.get_state().raw_sentence
            speak_text_live_instant(cur_raw, lang_code=LANG_TTS_VOICE.get(selected_lang, "en-US"))
        st.rerun()

    st.markdown("---")
    st.markdown("### Quick Guide")
    st.caption("1. Click **Start** to begin translation.")
    st.caption("2. Show gestures to camera or use instant triggers.")
    st.caption("3. Watch gestures accumulate into a sentence.")
    st.caption("4. Click **Improve** for AI grammar refinement.")
    st.caption("5. Click **Speak** to hear the sentence via gTTS.")


# ──────────────────────────────────────────────────────────────────────────────
# Header & Safety Emergency Monitor
# ──────────────────────────────────────────────────────────────────────────────

header_col, badge_col = st.columns([5, 1])
with header_col:
    st.markdown('<div class="app-header">🤟 VoiceSignAI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-sub">Multilingual Real-Time Sign Language → Sentence Structure → AI Refinement → Spoken Voice</div>',
        unsafe_allow_html=True,
    )
with badge_col:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="mode-badge badge-lens">● Ready</span>', unsafe_allow_html=True)

# Emergency SOS Alert Banner
state = pipeline.get_state()
if state.current_gesture == "help":
    st.markdown(
        """
        <div style="padding: 1rem 1.2rem; background: rgba(239, 68, 68, 0.2); border: 2px solid #ef4444; border-radius: 12px; margin-bottom: 1rem; color: #fca5a5; font-weight: 700; display: flex; align-items: center; justify-content: space-between;">
            <span>🚨 EMERGENCY SOS ACTIVE — Signer has presented the emergency HELP gesture!</span>
            <span class="mode-badge badge-demo" style="background:#ef4444;color:#fff;">PRIORITY ALERT</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")


# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────

lens_tab, custom_tab, dataset_tab = st.tabs([
    "🔍 Live Sign to Voice",
    "✍️ Custom Sign Studio",
    "📖 Multilingual Vocabulary",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: LIVE SIGN TO VOICE
# ══════════════════════════════════════════════════════════════════════════════

with lens_tab:

    # Active Language & Voice Banner
    d_col1, d_col2 = st.columns([2.5, 1])
    with d_col1:
        st.markdown(
            f"""
            <div style="background:rgba(30,41,59,0.85);border:1px solid rgba(56,189,248,0.35);border-radius:10px;padding:0.6rem 1rem;margin-bottom:0.8rem;">
                <b style="color:#38bdf8;">Active Sign Language:</b> <span style="color:#ffffff;font-weight:700;">{selected_lang}</span> &nbsp; • &nbsp; 
                <small style="color:#94a3b8;">Spoken Voice: <b>{LANG_TTS_VOICE.get(selected_lang, 'en-US')}</b></small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d_col2:
        st.markdown(
            f"""
            <div style="background:rgba(30,41,59,0.85);border:1px solid rgba(74,222,128,0.35);border-radius:10px;padding:0.6rem 1rem;margin-bottom:0.8rem;text-align:center;">
                <span style="color:#86efac;font-weight:700;">{'⚡ Live Auto-Voice ON' if auto_speech_active else '🔇 Live Auto-Voice OFF'}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Top controls ──────────────────────────────────────────────────────────
    ctrl_cols = st.columns([1, 1, 1, 1, 1, 1])

    with ctrl_cols[0]:
        if st.button("▶ Start", use_container_width=True, key="lens_start"):
            pipeline.start()
            st.rerun()

    with ctrl_cols[1]:
        if st.button("⏹ Stop", use_container_width=True, key="lens_stop"):
            pipeline.stop()
            st.rerun()

    with ctrl_cols[2]:
        if st.button("✨ Improve", use_container_width=True, key="lens_improve"):
            pipeline.improve_sentence()
            st.rerun()

    with ctrl_cols[3]:
        if st.button("🔊 Speak", use_container_width=True, key="lens_speak"):
            spoken = state.ai_enhanced_sentence if state.ai_enhanced_sentence else state.raw_sentence
            if spoken:
                speak_text_live_instant(spoken, lang_code=LANG_TTS_VOICE.get(selected_lang, "en-US"))
                pipeline.speak_sentence(use_ai=True)
            st.rerun()

    with ctrl_cols[4]:
        if st.button("⌫ Undo", use_container_width=True, key="lens_undo"):
            pipeline.remove_last_word()
            st.rerun()

    with ctrl_cols[5]:
        if st.button("🗑 Clear", use_container_width=True, key="lens_clear"):
            pipeline.clear_sentence()
            st.rerun()

    st.markdown("")

    # ── Demo mode button ──────────────────────────────────────────────────────
    if DEMO_MODE:
        if st.button(
            "🎬 Inject Demo Gestures",
            use_container_width=False,
            key="lens_demo",
            help="Injects scripted gestures: hello i need help",
        ):
            pipeline.run_demo_sequence()
            st.rerun()

    st.markdown("")

    # ── Main layout: camera | live gesture | sentence ─────────────────────────
    cam_col, info_col = st.columns([1.1, 1], gap="large")

    state = pipeline.get_state()

    with cam_col:
        st.markdown('<div class="card-title">Live Camera Feed</div>', unsafe_allow_html=True)

        cam_mode = st.radio(
            "Camera Input Mode",
            ["🎥 Live 30 FPS WebRTC Camera", "📷 High-Res Snapshot Analyzer"],
            horizontal=True,
            key="lens_cam_mode_choice",
            label_visibility="collapsed",
        )

        if cam_mode == "🎥 Live 30 FPS WebRTC Camera":
            voice_code = LANG_TTS_VOICE.get(selected_lang, "en-US")
            vocab_json = json.dumps(LANG_VOCAB_MAP.get(selected_lang, LANG_VOCAB_MAP["ASL (American)"]))
            is_active_js = "true" if state.translation_active else "false"

            webrtc_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
                <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>
                <style>
                    body {{ margin: 0; padding: 0; background: transparent; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #fff; overflow: hidden; }}
                    #container {{ position: relative; width: 100%; height: 440px; background: #07101f; border-radius: 14px; overflow: hidden; border: 2px solid #0284c7; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }}
                    #webcam {{ width: 100%; height: 100%; object-fit: cover; transform: scaleX(-1); display: none; }}
                    #canvas {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; transform: scaleX(-1); }}
                    
                    .top-badge {{ position: absolute; top: 12px; left: 12px; background: rgba(0,0,0,0.75); padding: 6px 14px; border-radius: 8px; font-size: 13px; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 8px; }}
                    .fps-badge {{ position: absolute; top: 12px; right: 12px; background: rgba(34,197,94,0.25); border: 1px solid #22c55e; padding: 5px 12px; border-radius: 6px; font-size: 12px; font-weight: 700; color: #86efac; }}
                    
                    .controls-bar {{ position: absolute; top: 56px; right: 12px; display: flex; gap: 6px; }}
                    .hud-btn {{ background: rgba(15,23,42,0.85); border: 1px solid rgba(148,163,184,0.4); color: #f8fafc; padding: 5px 10px; border-radius: 6px; font-size: 12px; font-weight: 700; cursor: pointer; backdrop-filter: blur(6px); transition: all 0.2s; }}
                    .hud-btn:hover {{ background: #0284c7; border-color: #38bdf8; }}
                    .hud-btn-active {{ background: #22c55e; border-color: #4ade80; color: #000; }}
                    
                    .bottom-bar {{ position: absolute; bottom: 12px; left: 12px; right: 12px; background: rgba(15,23,42,0.92); backdrop-filter: blur(12px); padding: 10px 16px; border-radius: 10px; border: 1px solid rgba(148,163,184,0.25); display: flex; flex-direction: column; gap: 6px; }}
                    .sign-row {{ display: flex; justify-content: space-between; align-items: center; }}
                    .sign-display {{ font-size: 20px; font-weight: 800; color: #38bdf8; }}
                    .sentence-display {{ font-size: 14px; color: #e2e8f0; font-weight: 600; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 4px; display: flex; justify-content: space-between; align-items: center; }}
                    .pulsing-dot {{ width: 8px; height: 8px; background: #22c55e; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px #22c55e; }}
                    .dot-paused {{ background: #eab308; box-shadow: 0 0 8px #eab308; }}
                </style>
            </head>
            <body>
                <div id="container">
                    <video id="webcam" autoplay playsinline muted></video>
                    <canvas id="canvas"></canvas>
                    
                    <div class="top-badge">
                        <span class="pulsing-dot" id="statusDot"></span>
                        <span id="statusText">30 FPS MediaPipe AI • {selected_lang}</span>
                    </div>
                    
                    <div class="fps-badge" id="fpsCounter">30 FPS LIVE</div>

                    <div class="controls-bar">
                        <button class="hud-btn" id="btnToggleTrans" onclick="toggleTranslation()">⏹ Stop</button>
                        <button class="hud-btn" onclick="speakCurrentSentence()">🔊 Speak</button>
                        <button class="hud-btn" onclick="clearSentence()">🗑 Clear</button>
                    </div>

                    <div class="bottom-bar">
                        <div class="sign-row">
                            <span>Detected: <b class="sign-display" id="liveSignLabel">—</b></span>
                            <span id="confLabel" style="color: #4ade80; font-size: 13px; font-weight: 700;">Waiting for hand...</span>
                        </div>
                        <div class="sentence-display">
                            <div><span>Live Sentence: </span><span id="liveSentence" style="color:#7dd3fc; font-style:italic;">Show signs to start speaking...</span></div>
                        </div>
                    </div>
                </div>

                <script>
                const vocab = {vocab_json};
                const ttsLang = "{voice_code}";
                let isTranslationActive = {is_active_js};

                const videoElement = document.getElementById('webcam');
                const canvasElement = document.getElementById('canvas');
                const canvasCtx = canvasElement.getContext('2d');
                const signLabel = document.getElementById('liveSignLabel');
                const confLabel = document.getElementById('confLabel');
                const sentenceLabel = document.getElementById('liveSentence');
                const fpsCounter = document.getElementById('fpsCounter');
                const btnToggle = document.getElementById('btnToggleTrans');
                const statusDot = document.getElementById('statusDot');
                const statusText = document.getElementById('statusText');

                let accumulatedWords = [];
                let lastAcceptedSign = "";
                let lastSpeechTime = 0;
                let stableCount = 0;
                let lastCandidate = "";
                let frameCount = 0;
                let lastFpsTime = performance.now();

                function updateUiStatus() {{
                    if (isTranslationActive) {{
                        btnToggle.innerText = "⏹ Stop";
                        btnToggle.style.background = "rgba(15,23,42,0.85)";
                        statusDot.className = "pulsing-dot";
                        statusText.innerText = "30 FPS MediaPipe AI • {selected_lang}";
                    }} else {{
                        btnToggle.innerText = "▶ Start";
                        btnToggle.style.background = "#22c55e";
                        btnToggle.style.color = "#000";
                        statusDot.className = "pulsing-dot dot-paused";
                        statusText.innerText = "⏸ Translation Paused";
                        signLabel.innerText = "PAUSED";
                        confLabel.innerText = "Click Start to resume";
                        if ('speechSynthesis' in window) {{
                            window.speechSynthesis.cancel();
                        }}
                    }}
                }}
                updateUiStatus();

                function toggleTranslation() {{
                    isTranslationActive = !isTranslationActive;
                    updateUiStatus();
                }}

                function clearSentence() {{
                    accumulatedWords = [];
                    lastAcceptedSign = "";
                    sentenceLabel.innerText = "Show signs to start speaking...";
                    signLabel.innerText = "—";
                }}

                function speakCurrentSentence() {{
                    if (accumulatedWords.length > 0) {{
                        speakNow(accumulatedWords.join(" "));
                    }}
                }}

                function speakNow(text) {{
                    if (!text || text.trim() === "") return;
                    if ('speechSynthesis' in window) {{
                        window.speechSynthesis.cancel();
                        const utt = new SpeechSynthesisUtterance(text);
                        utt.lang = ttsLang;
                        utt.rate = 1.0;
                        utt.pitch = 1.0;
                        window.speechSynthesis.speak(utt);
                    }}
                }}

                function dist(p1, p2) {{
                    return Math.hypot(p1.x - p2.x, p1.y - p2.y);
                }}

                function classifyLandmarks(landmarks) {{
                    if (!landmarks || landmarks.length < 21) return {{ sign: "hello", conf: 0.85 }};
                    const wrist = landmarks[0];
                    
                    function isExt(tipIdx, pipIdx) {{
                        return dist(landmarks[tipIdx], wrist) > dist(landmarks[pipIdx], wrist) * 1.15;
                    }}

                    const thumb = isExt(4, 2);
                    const index = isExt(8, 6);
                    const middle = isExt(12, 10);
                    const ring = isExt(16, 14);
                    const pinky = isExt(20, 18);

                    if (thumb && !index && !middle && !ring && !pinky) {{
                        if (landmarks[4].y < landmarks[2].y) {{
                            return {{ sign: vocab["yes"] || "yes", conf: 0.95 }};
                        }} else {{
                            return {{ sign: vocab["no"] || "no", conf: 0.92 }};
                        }}
                    }} else if (thumb && index && middle && ring && pinky) {{
                        return {{ sign: vocab["hello"] || "hello", conf: 0.96 }};
                    }} else if (!thumb && index && middle && ring && !pinky) {{
                        return {{ sign: vocab["water"] || "water", conf: 0.94 }};
                    }} else if (!thumb && !index && !middle && !ring && !pinky) {{
                        return {{ sign: vocab["need"] || "need", conf: 0.90 }};
                    }} else if (!thumb && index && !middle && !ring && !pinky) {{
                        return {{ sign: vocab["help"] || "help", conf: 0.93 }};
                    }} else if (thumb && !index && !middle && !ring && pinky) {{
                        return {{ sign: vocab["thank you"] || "thank you", conf: 0.91 }};
                    }} else if (index && middle && !ring && !pinky) {{
                        return {{ sign: vocab["please"] || "please", conf: 0.88 }};
                    }} else {{
                        return {{ sign: vocab["hello"] || "hello", conf: 0.88 }};
                    }}
                }}

                const CONNECTIONS = [
                    [0,1],[1,2],[2,3],[3,4],
                    [0,5],[5,6],[6,7],[7,8],
                    [5,9],[9,10],[10,11],[11,12],
                    [9,13],[13,14],[14,15],[15,16],
                    [13,17],[17,18],[18,19],[19,20],
                    [0,17],[5,9],[9,13],[13,17]
                ];

                function onResults(results) {{
                    frameCount++;
                    const now = performance.now();
                    if (now - lastFpsTime >= 1000) {{
                        const currentFps = Math.round((frameCount * 1000) / (now - lastFpsTime));
                        fpsCounter.innerText = currentFps + " FPS LIVE";
                        frameCount = 0;
                        lastFpsTime = now;
                    }}

                    canvasElement.width = videoElement.videoWidth || 640;
                    canvasElement.height = videoElement.videoHeight || 480;

                    canvasCtx.save();
                    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
                    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

                    if (!isTranslationActive) {{
                        canvasCtx.restore();
                        return;
                    }}

                    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {{
                        for (const landmarks of results.multiHandLandmarks) {{
                            canvasCtx.strokeStyle = "#00f078";
                            canvasCtx.lineWidth = 3;
                            for (const [i, j] of CONNECTIONS) {{
                                const p1 = landmarks[i];
                                const p2 = landmarks[j];
                                canvasCtx.beginPath();
                                canvasCtx.moveTo(p1.x * canvasElement.width, p1.y * canvasElement.height);
                                canvasCtx.lineTo(p2.x * canvasElement.width, p2.y * canvasElement.height);
                                canvasCtx.stroke();
                            }}

                            for (let idx = 0; idx < landmarks.length; idx++) {{
                                const p = landmarks[idx];
                                canvasCtx.beginPath();
                                canvasCtx.arc(p.x * canvasElement.width, p.y * canvasElement.height, [4,8,12,16,20].includes(idx) ? 7 : 5, 0, 2 * Math.PI);
                                canvasCtx.fillStyle = [4,8,12,16,20].includes(idx) ? "#38bdf8" : "#00ff80";
                                canvasCtx.fill();
                            }}

                            const res = classifyLandmarks(landmarks);
                            signLabel.innerText = res.sign.toUpperCase();
                            confLabel.innerText = Math.round(res.conf * 100) + "% Confidence";

                            if (res.sign === lastCandidate) {{
                                stableCount++;
                            }} else {{
                                lastCandidate = res.sign;
                                stableCount = 1;
                            }}

                            const currentTime = Date.now();
                            if (stableCount >= 4 && (res.sign !== lastAcceptedSign || (currentTime - lastSpeechTime > 2500))) {{
                                lastAcceptedSign = res.sign;
                                lastSpeechTime = currentTime;
                                accumulatedWords.push(res.sign);
                                if (accumulatedWords.length > 8) accumulatedWords.shift();

                                sentenceLabel.innerText = accumulatedWords.join(" ");
                                speakNow(res.sign);
                            }}
                        }}
                    }} else {{
                        signLabel.innerText = "—";
                        confLabel.innerText = "Show hand to camera...";
                    }}
                    canvasCtx.restore();
                }}

                const hands = new Hands({{
                    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${{file}}`
                }});

                hands.setOptions({{
                    maxNumHands: 2,
                    modelComplexity: 1,
                    minDetectionConfidence: 0.4,
                    minTrackingConfidence: 0.4
                }});

                hands.onResults(onResults);

                if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {{
                    const camera = new Camera(videoElement, {{
                        onFrame: async () => {{
                            await hands.send({{ image: videoElement }});
                        }},
                        width: 640,
                        height: 480
                    }});
                    camera.start();
                }}
                </script>
            </body>
            </html>
            """
            components.html(webrtc_html, height=460)
            st.caption("💡 Show your hand clearly to the live camera above. The AI tracks 21 hand landmarks and speaks each sign aloud in real time.")
        else:
            cam_image = st.camera_input("Visual Sign Input", key="lens_cam_stream", label_visibility="collapsed")
            if cam_image is not None:
                try:
                    from PIL import Image
                    pil_img = Image.open(cam_image)
                    rgb_arr = np.array(pil_img.convert("RGB"))
                    annotated_frame, sign_label, conf_val, num_hands = process_camera_frame(rgb_arr, language_choice=selected_lang)

                    st.image(annotated_frame, caption=f"MediaPipe 3D Landmark Tracking: {num_hands} hand(s) detected", use_container_width=True)

                    if sign_label and state.translation_active:
                        for _ in range(STABILITY_WINDOW):
                            pipeline.push_ml_prediction({"gesture": sign_label, "confidence": conf_val})

                        if auto_speech_active:
                            speak_text_live_instant(sign_label, lang_code=LANG_TTS_VOICE.get(selected_lang, "en-US"))

                        st.success(f"🎯 Recognized Sign: **{sign_label.upper()}** ({int(conf_val*100)}% confidence)")
                except Exception as e:
                    st.warning(f"Camera visualizer note: {e}")
            else:
                st.markdown(
                    """
                    <div class="camera-placeholder">
                        📷<br><br>
                        Camera snapshot active.<br>
                        <small>Capture a gesture with your webcam above or switch to Live 30 FPS mode.</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Translation status
        if state.translation_active:
            st.markdown(
                '🟢 <span class="status-active">Translation Active</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '⚫ <span class="status-inactive">Translation Stopped</span>',
                unsafe_allow_html=True,
            )

    with info_col:

        # ── Live gesture ──────────────────────────────────────────────────────
        st.markdown(
            '<div class="card">'
            '<div class="card-title">Live Detected Sign</div>',
            unsafe_allow_html=True,
        )

        if state.current_gesture:
            gesture_display = state.current_gesture.upper()
            conf_pct = int(state.gesture_confidence * 100)
            conf_color = "#56d364" if state.gesture_confidence >= 0.85 else \
                         "#f0b429" if state.gesture_confidence >= CONFIDENCE_THRESHOLD else "#f85149"

            st.markdown(
                f'<div class="gesture-live">{gesture_display}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="margin-top:0.4rem;font-size:0.82rem;color:#8b949e;">'
                f'Confidence: <span style="color:{conf_color};font-weight:700;">{conf_pct}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.progress(state.gesture_confidence)
        else:
            st.markdown(
                '<div class="gesture-live gesture-none">—</div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Raw sentence ──────────────────────────────────────────────────────
        st.markdown(
            '<div class="card">'
            '<div class="card-title">Recognised Sentence (Raw Sequence)</div>',
            unsafe_allow_html=True,
        )
        if state.raw_sentence:
            tokens_html = " ".join(f'<span class="token-chip">{w}</span>' for w in state.raw_sentence.split())
            st.markdown(
                f'<div class="raw-sentence" style="margin-top:0.3rem;">{tokens_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="raw-sentence sentence-empty">Show signs to build sentence tokens...</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── AI-enhanced sentence ──────────────────────────────────────────────
        st.markdown(
            '<div class="card">'
            '<div class="card-title">✨ Gemini 2.5 AI-Refined Sentence</div>',
            unsafe_allow_html=True,
        )
        if state.ai_enhanced_sentence:
            st.markdown(
                f'<div class="ai-sentence" style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 182, 212, 0.08) 100%); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 10px; padding: 0.85rem 1.1rem; box-shadow: 0 4px 16px rgba(16, 185, 129, 0.15);">{state.ai_enhanced_sentence}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="ai-sentence sentence-empty">'
                'Click Improve above to enhance structure with Gemini AI.'
                '</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Error display ─────────────────────────────────────────────────────
        if state.last_error:
            st.error(f"⚠️ {state.last_error}")

    # ── TTS audio playback ────────────────────────────────────────────────────
    play_audio_if_ready()

    # ── Reset ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    reset_col, _ = st.columns([1, 4])
    with reset_col:
        if st.button("🔄 Reset Session", key="lens_reset"):
            pipeline.reset_session()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: CUSTOM SIGN ENROLLMENT STUDIO ("TEACH MY SIGN")
# ══════════════════════════════════════════════════════════════════════════════
with custom_tab:
    st.markdown('<span class="mode-badge badge-active">✍️ CUSTOM SIGN STUDIO</span> &nbsp; <small style="color:#94a3b8;">Enroll both <b>Static Hand Poses</b> and <b>Dynamic Motion Sequences</b> directly into SQLite.</small>', unsafe_allow_html=True)
    st.markdown("")

    form_col, list_col = st.columns([1.1, 1.2], gap="large")

    with form_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Register New Custom Sign</div>', unsafe_allow_html=True)

        new_label = st.text_input("Sign Name / Word", placeholder="e.g. Akheel, Classroom, Doctor, HelpMe", key="cs_label_in")
        new_desc = st.text_input("Description / Notes", placeholder="e.g. Hand moving from chin to chest", key="cs_desc_in")

        sign_type_choice = st.radio(
            "Gesture Type",
            ["📸 Static Sign (Single Pose / Keyframe)", "🎬 Dynamic Sign (Continuous Motion Trajectory)"],
            key="cs_sign_type_radio",
            horizontal=True,
        )

        if "Static" in sign_type_choice:
            st.caption("📸 **Static Sign**: Hold a fixed hand pose in front of the camera.")
            custom_snap = st.camera_input("Capture Pose Snapshot", key="cs_cam_static_in")

            if st.button("💾 Save Static Sign to Database", key="cs_save_static_btn", use_container_width=True):
                if not new_label.strip():
                    st.error("Please provide a sign name.")
                elif custom_snap is None:
                    st.error("Please capture a hand gesture snapshot using the camera.")
                else:
                    try:
                        pil_img_c = Image.open(custom_snap)
                        rgb_arr_c = np.array(pil_img_c.convert("RGB"))
                        detector = load_hand_landmarker()
                        if detector:
                            import mediapipe as mp
                            mp_img_c = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_arr_c)
                            res_c = detector.detect(mp_img_c)
                            if res_c.hand_landmarks:
                                lms = res_c.hand_landmarks[0]
                                pts = []
                                for lm in lms:
                                    pts.extend([float(lm.x), float(lm.y), float(lm.z)])

                                sample_in = CustomSignSampleInput(
                                    sample_type=SampleTypeEnum.STATIC,
                                    features=pts,
                                    motion_energy=0.01,
                                )
                                sign_id = f"sign_{int(time.time())}"
                                create_custom_sign(
                                    sign_id=sign_id,
                                    user_id="default_user",
                                    label=new_label.strip(),
                                    description=new_desc.strip(),
                                    samples=[sample_in],
                                )
                                st.success(f"✅ Static Sign **{new_label.strip()}** successfully registered in SQLite database!")
                            else:
                                st.error("No hand detected in the snapshot. Please hold your hand clearly in frame.")
                    except Exception as e:
                        st.error(f"Error saving custom sign: {e}")
        else:
            st.caption("🎬 **Dynamic Sign**: Record a live 30 FPS motion clip with Start/Stop or upload a video file.")
            dyn_method = st.radio("Dynamic Input Method", ["🎥 Live 30 FPS Motion Clip Recorder", "📁 Upload Video Gesture Clip (.mp4/.mov)"], horizontal=True, key="cs_dyn_method")

            if dyn_method == "🎥 Live 30 FPS Motion Clip Recorder":
                st.caption("🔴 Click **'Start Recording'** on the camera to capture a 30 FPS continuous gesture motion sequence:")
                rec_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
                    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>
                    <style>
                        body {{ margin: 0; padding: 0; background: transparent; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #fff; overflow: hidden; }}
                        #container {{ position: relative; width: 100%; height: 380px; background: #07101f; border-radius: 14px; overflow: hidden; border: 2px solid #a855f7; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }}
                        #webcam {{ width: 100%; height: 100%; object-fit: cover; transform: scaleX(-1); display: none; }}
                        #canvas {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; transform: scaleX(-1); }}
                        
                        .top-badge {{ position: absolute; top: 12px; left: 12px; background: rgba(0,0,0,0.75); padding: 5px 12px; border-radius: 8px; font-size: 13px; font-weight: 700; color: #c084fc; }}
                        .btn-group {{ position: absolute; top: 12px; right: 12px; display: flex; gap: 8px; }}
                        .rec-btn {{ background: #ef4444; border: 0; color: #fff; padding: 6px 14px; border-radius: 6px; font-size: 13px; font-weight: 700; cursor: pointer; }}
                        .stop-btn {{ background: #3b82f6; border: 0; color: #fff; padding: 6px 14px; border-radius: 6px; font-size: 13px; font-weight: 700; cursor: pointer; display: none; }}
                        
                        .bottom-bar {{ position: absolute; bottom: 12px; left: 12px; right: 12px; background: rgba(15,23,42,0.92); backdrop-filter: blur(8px); padding: 8px 14px; border-radius: 8px; border: 1px solid rgba(148,163,184,0.25); display: flex; justify-content: space-between; align-items: center; }}
                    </style>
                </head>
                <body>
                    <div id="container">
                        <video id="webcam" autoplay playsinline muted></video>
                        <canvas id="canvas"></canvas>
                        
                        <div class="top-badge">
                            <span>🎬 30 FPS Dynamic Motion Recorder</span>
                        </div>

                        <div class="btn-group">
                            <button class="rec-btn" id="btnStart" onclick="startRec()">🔴 Start Recording</button>
                            <button class="stop-btn" id="btnStop" onclick="stopRec()">⏹ Stop & Finish Clip</button>
                        </div>

                        <div class="bottom-bar">
                            <span id="statusTxt" style="font-weight: 600; font-size: 13px; color: #e2e8f0;">Ready. Click Start Recording to capture dynamic motion.</span>
                            <span id="counterTxt" style="color: #c084fc; font-weight: 700; font-size: 13px;">0 / 30 Frames</span>
                        </div>
                    </div>

                    <script>
                    const videoElement = document.getElementById('webcam');
                    const canvasElement = document.getElementById('canvas');
                    const canvasCtx = canvasElement.getContext('2d');
                    const btnStart = document.getElementById('btnStart');
                    const btnStop = document.getElementById('btnStop');
                    const statusTxt = document.getElementById('statusTxt');
                    const counterTxt = document.getElementById('counterTxt');

                    let isRecording = false;
                    let recordedFrames = [];
                    const TARGET_FRAMES = 30;

                    function startRec() {{
                        recordedFrames = [];
                        isRecording = true;
                        btnStart.style.display = 'none';
                        btnStop.style.display = 'inline-block';
                        statusTxt.innerText = '🔴 Recording gesture trajectory... Perform motion now!';
                        statusTxt.style.color = '#f87171';
                    }}

                    function stopRec() {{
                        isRecording = false;
                        btnStart.style.display = 'inline-block';
                        btnStop.style.display = 'none';
                        statusTxt.innerText = '✅ Motion sequence captured (' + recordedFrames.length + ' frames)! Paste into Save box below.';
                        statusTxt.style.color = '#4ade80';
                        
                        // Copy JSON to clipboard for seamless 1-click paste
                        const jsonStr = JSON.stringify(recordedFrames);
                        navigator.clipboard.writeText(jsonStr).catch(() => {{}});
                    }}

                    const CONNECTIONS = [
                        [0,1],[1,2],[2,3],[3,4],
                        [0,5],[5,6],[6,7],[7,8],
                        [5,9],[9,10],[10,11],[11,12],
                        [9,13],[13,14],[14,15],[15,16],
                        [13,17],[17,18],[18,19],[19,20],
                        [0,17],[5,9],[9,13],[13,17]
                    ];

                    function onResults(results) {{
                        canvasElement.width = videoElement.videoWidth || 640;
                        canvasElement.height = videoElement.videoHeight || 480;

                        canvasCtx.save();
                        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
                        canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

                        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {{
                            const landmarks = results.multiHandLandmarks[0];
                            canvasCtx.strokeStyle = isRecording ? "#ef4444" : "#a855f7";
                            canvasCtx.lineWidth = 3;
                            for (const [i, j] of CONNECTIONS) {{
                                const p1 = landmarks[i];
                                const p2 = landmarks[j];
                                canvasCtx.beginPath();
                                canvasCtx.moveTo(p1.x * canvasElement.width, p1.y * canvasElement.height);
                                canvasCtx.lineTo(p2.x * canvasElement.width, p2.y * canvasElement.height);
                                canvasCtx.stroke();
                            }}

                            for (let idx = 0; idx < landmarks.length; idx++) {{
                                const p = landmarks[idx];
                                canvasCtx.beginPath();
                                canvasCtx.arc(p.x * canvasElement.width, p.y * canvasElement.height, 5, 0, 2 * Math.PI);
                                canvasCtx.fillStyle = isRecording ? "#f87171" : "#c084fc";
                                canvasCtx.fill();
                            }}

                            if (isRecording) {{
                                const pts = [];
                                for (const lm of landmarks) {{
                                    pts.push(lm.x, lm.y, lm.z);
                                }}
                                recordedFrames.push(pts);
                                counterTxt.innerText = recordedFrames.length + " / " + TARGET_FRAMES + " Frames";

                                if (recordedFrames.length >= TARGET_FRAMES) {{
                                    stopRec();
                                }}
                            }}
                        }}
                        canvasCtx.restore();
                    }}

                    const hands = new Hands({{
                        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${{file}}`
                    }});

                    hands.setOptions({{
                        maxNumHands: 1,
                        modelComplexity: 1,
                        minDetectionConfidence: 0.4,
                        minTrackingConfidence: 0.4
                    }});

                    hands.onResults(onResults);

                    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {{
                        const camera = new Camera(videoElement, {{
                            onFrame: async () => {{
                                await hands.send({{ image: videoElement }});
                            }},
                            width: 640,
                            height: 480
                        }});
                        camera.start();
                    }}
                    </script>
                </body>
                </html>
                """
                components.html(rec_html, height=400)

                motion_payload = st.text_area("Recorded Motion JSON (Auto-copied to clipboard upon recording)", placeholder="Click 'Start Recording' on the camera above to capture 30 frames, then paste here (Ctrl+V)...", key="cs_dyn_rec_json_in")

                if st.button("💾 Save Live Recorded Dynamic Sign", key="cs_save_dyn_rec_btn", use_container_width=True):
                    if not new_label.strip():
                        st.error("Please provide a sign name.")
                    elif not motion_payload.strip():
                        st.error("Please record a 30 FPS motion clip using the camera above and paste the JSON.")
                    else:
                        try:
                            frames_data = json.loads(motion_payload.strip())
                            if isinstance(frames_data, list) and len(frames_data) > 0:
                                energy = 0.0
                                for t in range(1, len(frames_data)):
                                    p_prev = np.array(frames_data[t - 1])
                                    p_curr = np.array(frames_data[t])
                                    energy += float(np.linalg.norm(p_curr - p_prev))
                                energy /= max(1, len(frames_data) - 1)

                                sample_in = CustomSignSampleInput(
                                    sample_type=SampleTypeEnum.DYNAMIC,
                                    frames=frames_data,
                                    motion_energy=energy,
                                )
                                sign_id = f"dyn_sign_{int(time.time())}"
                                create_custom_sign(
                                    sign_id=sign_id,
                                    user_id="default_user",
                                    label=new_label.strip(),
                                    description=new_desc.strip(),
                                    samples=[sample_in],
                                )
                                st.success(f"✅ Dynamic Sign **{new_label.strip()}** saved! ({len(frames_data)} frames, motion energy: {energy:.3f})")
                                st.rerun()
                            else:
                                st.error("Invalid frame sequence. Please record again.")
                        except Exception as e:
                            st.error(f"Error saving dynamic sign: {e}")

            else:
                video_file = st.file_uploader("Upload Gesture Video (.mp4, .mov)", type=["mp4", "mov", "avi"], key="cs_dyn_video_upload")
                if st.button("💾 Process & Save Uploaded Dynamic Sign", key="cs_save_dyn_video_btn", use_container_width=True):
                    if not new_label.strip():
                        st.error("Please provide a sign name.")
                    elif video_file is None:
                        st.error("Please upload a video file showing the dynamic gesture.")
                    else:
                        with st.spinner("Extracting 30 FPS 3D Landmark Trajectory with MediaPipe..."):
                            v_bytes = video_file.read()
                            seq_frames, energy = extract_video_landmark_sequence(v_bytes, target_frames=30)
                            if seq_frames and len(seq_frames) > 0:
                                sample_in = CustomSignSampleInput(
                                    sample_type=SampleTypeEnum.DYNAMIC,
                                    frames=seq_frames,
                                    motion_energy=energy,
                                )
                                sign_id = f"dyn_sign_{int(time.time())}"
                                create_custom_sign(
                                    sign_id=sign_id,
                                    user_id="default_user",
                                    label=new_label.strip(),
                                    description=new_desc.strip(),
                                    samples=[sample_in],
                                )
                                st.success(f"✅ Dynamic Sign **{new_label.strip()}** saved! ({len(seq_frames)} frames, motion energy: {energy:.3f})")
                                st.rerun()
                            else:
                                st.error("Could not extract hand landmarks from the video. Please ensure hands are clearly visible throughout the clip.")

        st.markdown("</div>", unsafe_allow_html=True)

    with list_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Enrolled Personal Vocabulary (SQLite)</div>', unsafe_allow_html=True)

        user_signs = list_custom_signs()
        if user_signs:
            for s in user_signs:
                s_col1, s_col2 = st.columns([3, 1.2])
                with s_col1:
                    type_badge = "🎬 DYNAMIC" if s.sample_type_summary == "dynamic" else "📸 STATIC"
                    badge_color = "#a78bfa" if s.sample_type_summary == "dynamic" else "#38bdf8"
                    st.markdown(
                        f"""
                        <div style="margin-bottom:0.25rem;">
                            <span style="font-size:1.15rem; font-weight:700; color:#f8fafc;">{s.label}</span>
                            &nbsp;<span style="background:rgba(255,255,255,0.08); border:1px solid {badge_color}; color:{badge_color}; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:700;">{type_badge}</span>
                        </div>
                        <small style="color:#94a3b8;">{s.description or 'Personal custom sign'} • {len(s.samples)} sample(s)</small>
                        """,
                        unsafe_allow_html=True,
                    )
                with s_col2:
                    c_b1, c_b2 = st.columns(2)
                    with c_b1:
                        if st.button("🔊", key=f"speak_cs_{s.id}", help="Test Voice Output"):
                            speak_text_live_instant(s.label, lang_code=LANG_TTS_VOICE.get(selected_lang, "en-US"))
                    with c_b2:
                        if st.button("🗑", key=f"del_{s.id}", help="Delete Sign"):
                            delete_custom_sign(s.id)
                            st.rerun()
                st.markdown("<hr style='margin: 0.4rem 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
        else:
            st.info("No custom signs registered yet. Use the enrollment form on the left to teach the AI static poses or dynamic motions!")

        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: MULTILINGUAL VOCABULARY & DICTIONARY
# ══════════════════════════════════════════════════════════════════════════════
with dataset_tab:
    st.markdown('<span class="mode-badge badge-active">📖 MULTILINGUAL VOCABULARY</span> &nbsp; <small style="color:#94a3b8;">Explore recognized sign vocabulary across ASL, ISL (Indian), and BSL (British).</small>', unsafe_allow_html=True)
    st.markdown("")

    # Dataset Metric Cards
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size: 2rem;">🇺🇸</div>
            <div style="font-weight: 700; font-size: 1.1rem; color:#79c0ff;">ASL (American)</div>
            <small style="color: #38bdf8;">American Sign Language</small>
            <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #94a3b8;">Vocabulary: Hello, Yes, No, Water, Need, Help, Thank You, Please</div>
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size: 2rem;">🇮🇳</div>
            <div style="font-weight: 700; font-size: 1.1rem; color:#79c0ff;">ISL (Indian)</div>
            <small style="color: #38bdf8;">Indian Sign Language</small>
            <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #94a3b8;">Vocabulary: Namaste, Agree, Disagree, Water, Need, Help, Dhanyawad, Kripya</div>
        </div>
        """, unsafe_allow_html=True)
    with d3:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size: 2rem;">🇬🇧</div>
            <div style="font-weight: 700; font-size: 1.1rem; color:#79c0ff;">BSL (British)</div>
            <small style="color: #38bdf8;">British Sign Language</small>
            <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #94a3b8;">Vocabulary: Hello, Yes, No, Water, Need, Help, Cheers, Please</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📖 Core Supported Vocabulary Guide")

    for row_start in range(0, len(SUPPORTED_SIGNS), 4):
        row_cols = st.columns(4)
        for col_idx, (icon, name, desc) in enumerate(SUPPORTED_SIGNS[row_start : row_start + 4]):
            with row_cols[col_idx]:
                st.markdown(
                    f"""
                    <div class="card" style="min-height: 130px; text-align: left;">
                        <div style="font-size: 2rem; margin-bottom: 0.3rem;">{icon}</div>
                        <div style="font-weight: 700; font-size: 1.05rem; color: #79c0ff;">{name}</div>
                        <div style="font-size: 0.84rem; color: #8b949e; margin-top: 0.3rem;">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )





st.markdown(
    """
    <div style="border-top: 1px solid #30363d; margin-top: 2.5rem; padding-top: 1rem; text-align: center; color: #8b949e; font-size: 0.85rem;">
        VoiceSignAI • Rebuilt Full-Stack UI • MediaPipe 3D Vision, Custom Sign Studio & Gemini 2.5 Flash Active
    </div>
    """,
    unsafe_allow_html=True,
)

