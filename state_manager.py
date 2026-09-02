"""
backend/state_manager.py — Sign2Voice
=======================================
Central, thread-safe application state.

All modules read/write state through this object.
Frontend reads state via to_dict() — no internal details exposed.
"""

import logging
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

# Valid speech status values
SPEECH_IDLE = "idle"
SPEECH_GENERATING = "generating"
SPEECH_PLAYING = "playing"
SPEECH_ERROR = "error"


@dataclass
class AppState:
    """
    The complete application state snapshot.

    raw_sentence and ai_enhanced_sentence are kept strictly separate.
    """
    current_gesture: str = ""
    gesture_confidence: float = 0.0
    raw_sentence: str = ""
    ai_enhanced_sentence: str = ""
    translation_active: bool = False
    speech_status: str = SPEECH_IDLE
    last_error: Optional[str] = None

    # Internal — not exposed to frontend
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def update(self, **kwargs) -> None:
        """Thread-safe field update. Only updates recognised fields."""
        valid_fields = {f for f in self.__dataclass_fields__ if not f.startswith("_")}
        with self._lock:
            for key, value in kwargs.items():
                if key in valid_fields:
                    setattr(self, key, value)
                else:
                    logger.warning("AppState.update: unknown field '%s' ignored.", key)

    def clear_error(self) -> None:
        with self._lock:
            self.last_error = None

    def set_error(self, message: str) -> None:
        with self._lock:
            self.last_error = message
        logger.error("AppState error: %s", message)

    def to_dict(self) -> dict:
        """
        Serialise to a plain dict safe for JSON / frontend consumption.
        Internal fields (prefixed _) are excluded.
        """
        return {
            "current_gesture": self.current_gesture,
            "gesture_confidence": round(self.gesture_confidence, 4),
            "raw_sentence": self.raw_sentence,
            "ai_enhanced_sentence": self.ai_enhanced_sentence,
            "translation_active": self.translation_active,
            "speech_status": self.speech_status,
            "last_error": self.last_error,
        }

    def reset(self) -> None:
        """Reset to default state (keeps lock instance)."""
        with self._lock:
            self.current_gesture = ""
            self.gesture_confidence = 0.0
            self.raw_sentence = ""
            self.ai_enhanced_sentence = ""
            self.translation_active = False
            self.speech_status = SPEECH_IDLE
            self.last_error = None
        logger.debug("AppState fully reset.")
