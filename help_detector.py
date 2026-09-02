"""
src/recognition/help_detector.py
--------------------------------
Safety & Accessibility Help Detection Monitor.
Tracks continuous recognition of the 'HELP' / 'EMERGENCY' sign over a persistent 5.0s window.
"""

from typing import Optional, Tuple
import time

from src.config import HELP_PERSISTENCE_SECONDS
from src.models.schemas import EventTypeEnum, RecognitionEvent, RecognitionResult


class HelpDetector:
    HELP_KEYWORDS = {"help", "emergency", "assistance", "need help", "danger", "sos"}

    def __init__(
        self,
        persistence_seconds: float = HELP_PERSISTENCE_SECONDS,
        grace_period_seconds: float = 0.6,
    ):
        self.persistence_seconds = persistence_seconds
        self.grace_period_seconds = grace_period_seconds
        self.first_help_detected_time: Optional[float] = None
        self.last_help_detected_time: Optional[float] = None
        self.is_alarm_triggered: bool = False

    def is_help_sign(self, label: str) -> bool:
        if not label:
            return False
        clean = label.strip().lower()
        return clean in self.HELP_KEYWORDS or any(kw in clean for kw in self.HELP_KEYWORDS)

    def process(self, recognition: RecognitionResult) -> Tuple[bool, float, Optional[RecognitionEvent]]:
        now = time.time()
        is_help = self.is_help_sign(recognition.label)

        if is_help:
            if self.first_help_detected_time is None:
                self.first_help_detected_time = now
                self.last_help_detected_time = now
                self.is_alarm_triggered = False
            else:
                self.last_help_detected_time = now

            duration = round(now - self.first_help_detected_time, 2)

            if duration >= self.persistence_seconds:
                self.is_alarm_triggered = True
                event = RecognitionEvent(
                    event=EventTypeEnum.HELP_DETECTED,
                    label=recognition.label,
                    language=recognition.language,
                    confidence=recognition.confidence,
                    sample_type=recognition.sample_type,
                    help_active=True,
                    help_duration_seconds=duration,
                    timestamp=now,
                )
                return True, duration, event

            return True, duration, None

        else:
            if self.last_help_detected_time is not None:
                elapsed_since_last = now - self.last_help_detected_time
                if elapsed_since_last > self.grace_period_seconds:
                    self.first_help_detected_time = None
                    self.last_help_detected_time = None
                    self.is_alarm_triggered = False
                    return False, 0.0, None
                else:
                    duration = round(now - self.first_help_detected_time, 2)
                    return True, duration, None

            return False, 0.0, None

    def reset(self) -> None:
        self.first_help_detected_time = None
        self.last_help_detected_time = None
        self.is_alarm_triggered = False


_GLOBAL_HELP_DETECTOR: Optional[HelpDetector] = None


def get_global_help_detector() -> HelpDetector:
    global _GLOBAL_HELP_DETECTOR
    if _GLOBAL_HELP_DETECTOR is None:
        _GLOBAL_HELP_DETECTOR = HelpDetector()
    return _GLOBAL_HELP_DETECTOR