"""
src/config.py
-------------
Central configuration for Multilingual & Personalized Sign Language System.
Manages data paths, model paths, landmark parameters, thresholds, and environment variables.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ASL_DATA_DIR = DATA_DIR / "asl"
ISL_DATA_DIR = DATA_DIR / "isl"
BSL_DATA_DIR = DATA_DIR / "bsl"
CUSTOM_DATA_DIR = DATA_DIR / "custom"
STATIC_DIR = BASE_DIR / "static"

DB_PATH = DATA_DIR / "sign_system.db"

STATIC_MODEL_PATH = BASE_DIR / "static_gesture_model.pkl"
DYNAMIC_MODEL_PATH = BASE_DIR / "dynamic_gesture_model.pkl"

LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
FEATURES_PER_HAND = LANDMARKS_PER_HAND * COORDS_PER_LANDMARK  # 63
TOTAL_FRAME_FEATURES = FEATURES_PER_HAND * 2  # 126
SEQUENCE_LENGTH = 30
TOTAL_DYNAMIC_FEATURES = SEQUENCE_LENGTH * TOTAL_FRAME_FEATURES  # 3780

MOTION_ENERGY_THRESHOLD = 0.038
CONFIDENCE_THRESHOLD = 0.65
HELP_PERSISTENCE_SECONDS = 5.0
DEBOUNCE_INTERVAL_SECONDS = 1.2

MIN_CUSTOM_SAMPLES_REQUIRED = 3
RECOMMENDED_CUSTOM_SAMPLES = 5

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")

SUPPORTED_LANGUAGES = ["ASL", "ISL", "BSL", "CUSTOM"]

for p in (DATA_DIR, ASL_DATA_DIR, ISL_DATA_DIR, BSL_DATA_DIR, CUSTOM_DATA_DIR):
    p.mkdir(parents=True, exist_ok=True)
