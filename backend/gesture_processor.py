"""
backend/gesture_processor.py — Sign2Voice
==========================================
Converts a raw stream of ML predictions into a stable stream of accepted gestures.

Pipeline:
  ML frame → confidence check → stability window → cooldown → accepted gesture

All thresholds are read from config.py; nothing is hardcoded here.
"""

import time
import logging
from collections import deque
from typing import Optional

from config import CONFIDENCE_THRESHOLD, STABILITY_WINDOW, GESTURE_COOLDOWN

logger = logging.getLogger(__name__)


class GestureProcessor:
    """
    Accepts or rejects per-frame ML predictions.

    Rules:
    1. Confidence below CONFIDENCE_THRESHOLD  → rejected immediately.
    2. Gesture must appear in STABILITY_WINDOW consecutive frames → accepted.
    3. After acceptance, GESTURE_COOLDOWN seconds must pass before the next
       gesture is accepted (prevents one sign from flooding the sentence).
    4. Consecutive duplicate gestures entering the sentence are deduplicated
       (e.g. hello, hello → hello).
    """

    def __init__(self) -> None:
        self._window: deque[str] = deque(maxlen=STABILITY_WINDOW)
        self._last_accepted_time: float = 0.0
        self._last_accepted_gesture: str = ""
        self._total_accepted: int = 0
        self._total_rejected: int = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def process(self, gesture: str, confidence: float) -> Optional[str]:
        """
        Feed one ML prediction frame.

        Args:
            gesture:    Gesture label (e.g. "hello").
            confidence: Model confidence in [0.0, 1.0].

        Returns:
            The accepted gesture string if all criteria are met, else None.
        """
        gesture = gesture.strip().lower()
        if not gesture:
            return None

        # ── 1. Confidence gate ────────────────────────────────────────────────
        if confidence < CONFIDENCE_THRESHOLD:
            logger.debug(
                "Gesture '%s' rejected: confidence %.2f < threshold %.2f",
                gesture, confidence, CONFIDENCE_THRESHOLD,
            )
            self._total_rejected += 1
            self._window.clear()          # reset window on low-confidence frame
            return None

        # ── 2. Stability window ───────────────────────────────────────────────
        self._window.append(gesture)

        # Window not yet full → wait for more frames
        if len(self._window) < STABILITY_WINDOW:
            return None

        # All entries must be the same gesture
        if len(set(self._window)) != 1:
            return None

        # ── 3. Cooldown ───────────────────────────────────────────────────────
        now = time.monotonic()
        elapsed = now - self._last_accepted_time
        if elapsed < GESTURE_COOLDOWN:
            logger.debug(
                "Gesture '%s' suppressed: cooldown %.2fs remaining",
                gesture, GESTURE_COOLDOWN - elapsed,
            )
            return None

        # ── 4. Consecutive-duplicate dedup ────────────────────────────────────
        if gesture == self._last_accepted_gesture:
            logger.debug(
                "Gesture '%s' suppressed: same as last accepted gesture",
                gesture,
            )
            return None

        # ── Accept ────────────────────────────────────────────────────────────
        self._last_accepted_time = now
        self._last_accepted_gesture = gesture
        self._total_accepted += 1
        self._window.clear()              # reset window after acceptance

        logger.info(
            "Gesture accepted: '%s' (conf=%.2f, total_accepted=%d)",
            gesture, confidence, self._total_accepted,
        )
        return gesture

    def reset(self) -> None:
        """Clear all internal state (e.g. on session reset)."""
        self._window.clear()
        self._last_accepted_time = 0.0
        self._last_accepted_gesture = ""
        logger.debug("GestureProcessor reset.")

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "total_accepted": self._total_accepted,
            "total_rejected": self._total_rejected,
            "last_accepted_gesture": self._last_accepted_gesture,
            "window_contents": list(self._window),
        }
