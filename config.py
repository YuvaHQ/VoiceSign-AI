"""
config.py — Sign2Voice
======================
Single source of truth for all configurable constants.
All modules import from here; nothing is hardcoded elsewhere.
"""

import os
from dotenv import load_dotenv

# Load .env from the project root (silently ignores missing file)
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# GESTURE PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

# Minimum ML confidence score to even consider a gesture (0.0 – 1.0)
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))

# Number of consecutive identical predictions required before accepting
STABILITY_WINDOW: int = int(os.getenv("STABILITY_WINDOW", "3"))

# Seconds that must pass before the next gesture can be accepted
GESTURE_COOLDOWN: float = float(os.getenv("GESTURE_COOLDOWN", "1.5"))

# ──────────────────────────────────────────────────────────────────────────────
# OPENAI
# ──────────────────────────────────────────────────────────────────────────────

# Never hardcode keys — always read from environment
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

# Model used for sentence improvement (cheap + fast for hackathon)
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Seconds before an OpenAI request is considered timed out
OPENAI_TIMEOUT: float = float(os.getenv("OPENAI_TIMEOUT", "10.0"))

# ──────────────────────────────────────────────────────────────────────────────
# TEXT-TO-SPEECH  (gTTS)
# ──────────────────────────────────────────────────────────────────────────────

# BCP-47 language code for gTTS
TTS_LANGUAGE: str = os.getenv("TTS_LANGUAGE", "en")

# Use gTTS slow mode (clearer but slower)
TTS_SLOW: bool = os.getenv("TTS_SLOW", "false").lower() == "true"

# ──────────────────────────────────────────────────────────────────────────────
# ML & ASSETS
# ──────────────────────────────────────────────────────────────────────────────

STATIC_MODEL_PATH: str = os.getenv("STATIC_MODEL_PATH", "static_gesture_model.pkl")
DYNAMIC_MODEL_PATH: str = os.getenv("DYNAMIC_MODEL_PATH", "dynamic_gesture_model.pkl")
HAND_LANDMARKER_PATH: str = os.getenv("HAND_LANDMARKER_PATH", "hand_landmarker.task")

# Supported signs metadata for reference and UI display
SUPPORTED_SIGNS: list[tuple[str, str, str]] = [
    ("👋", "Hello", "A warm greeting to start a conversation."),
    ("👍", "Yes", "Confirm, agree, or acknowledge clearly."),
    ("👎", "No", "Politely communicate a negative response."),
    ("🙏", "Thank You", "Express appreciation with confidence."),
    ("🤲", "Please", "Make a courteous request."),
    ("🆘", "Help", "Ask for assistance when it matters most."),
    ("💧", "Water", "Request water or hydration."),
    ("✋", "Need", "Express necessity or requirement."),
]

# ──────────────────────────────────────────────────────────────────────────────
# DEMO / MOCK MODE
# ──────────────────────────────────────────────────────────────────────────────

# When True, a scripted gesture sequence is injected instead of live ML input.
DEMO_MODE: bool = os.getenv("DEMO_MODE", "false").lower() == "true"

# Pre-scripted gestures for demo mode (repeats simulate stabilisation frames)
DEMO_GESTURE_SEQUENCE: list[dict] = [
    {"gesture": "hello", "confidence": 0.95},
    {"gesture": "hello", "confidence": 0.96},
    {"gesture": "hello", "confidence": 0.97},
    {"gesture": "i",     "confidence": 0.91},
    {"gesture": "i",     "confidence": 0.92},
    {"gesture": "i",     "confidence": 0.93},
    {"gesture": "need",  "confidence": 0.88},
    {"gesture": "need",  "confidence": 0.89},
    {"gesture": "need",  "confidence": 0.90},
    {"gesture": "help",  "confidence": 0.93},
    {"gesture": "help",  "confidence": 0.94},
    {"gesture": "help",  "confidence": 0.95},
]

