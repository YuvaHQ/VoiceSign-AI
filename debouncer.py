"""
src/recognition/debouncer.py
----------------------------
Meeting Mode recognition debouncer and transcript aggregator.
"""

from typing import List, Optional, Tuple
import time

from src.config import DEBOUNCE_INTERVAL_SECONDS
from src.models.schemas import EventTypeEnum, RecognitionEvent, RecognitionResult


class MeetingDebouncer:
    def __init__(
        self,
        debounce_interval: float = DEBOUNCE_INTERVAL_SECONDS,
        confidence_threshold: float = 0.60,
    ):
        self.debounce_interval = debounce_interval
        self.confidence_threshold = confidence_threshold
        self.last_added_word: Optional[str] = None
        self.last_added_time: float = 0.0
        self.words: List[str] = []
        self.is_paused: bool = False

    def process(self, recognition: RecognitionResult) -> Tuple[bool, Optional[RecognitionEvent]]:
        if self.is_paused:
            return False, None

        if recognition.confidence < self.confidence_threshold:
            return False, None

        label = recognition.label.strip()
        if not label or label.lower() in ("buffering...", "none", "unknown"):
            return False, None

        now = time.time()
        should_append = False

        if self.last_added_word is None:
            should_append = True
        elif label.lower() != self.last_added_word.lower():
            should_append = True
        elif (now - self.last_added_time) >= self.debounce_interval:
            should_append = True

        if should_append:
            self.words.append(label)
            self.last_added_word = label
            self.last_added_time = now

            event = RecognitionEvent(
                event=EventTypeEnum.TRANSCRIPT_UPDATED,
                label=label,
                language=recognition.language,
                confidence=recognition.confidence,
                sample_type=recognition.sample_type,
                is_custom=recognition.is_custom,
                transcript=self.get_transcript(),
                recent_words=self.get_recent_words(count=5),
                timestamp=now,
            )
            return True, event

        return False, None

    def get_transcript(self) -> str:
        return " ".join(self.words)

    def get_recent_words(self, count: int = 5) -> List[str]:
        return self.words[-count:] if self.words else []

    def clear(self) -> None:
        self.words.clear()
        self.last_added_word = None
        self.last_added_time = 0.0

    def pause(self) -> None:
        self.is_paused = True

    def resume(self) -> None:
        self.is_paused = False


_GLOBAL_DEBOUNCER: Optional[MeetingDebouncer] = None


def get_global_debouncer() -> MeetingDebouncer:
    global _GLOBAL_DEBOUNCER
    if _GLOBAL_DEBOUNCER is None:
        _GLOBAL_DEBOUNCER = MeetingDebouncer()
    return _GLOBAL_DEBOUNCER