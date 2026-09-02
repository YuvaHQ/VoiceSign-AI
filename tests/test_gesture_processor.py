"""
tests/test_gesture_processor.py — Sign2Voice
=============================================
Unit tests for GestureProcessor.

Covers: confidence filtering, stability window, cooldown, dedup.
Uses time.monotonic patching for cooldown tests — no real sleeping.
"""

import time
import pytest
from unittest.mock import patch

# Ensure project root is on path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.gesture_processor import GestureProcessor


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def feed_n(processor: GestureProcessor, gesture: str, confidence: float, n: int):
    """Feed the same gesture n times and return list of non-None results."""
    results = []
    for _ in range(n):
        r = processor.process(gesture, confidence)
        if r is not None:
            results.append(r)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 1. Single gesture — accepted after stability window
# ──────────────────────────────────────────────────────────────────────────────

def test_single_gesture_accepted():
    """A gesture appearing STABILITY_WINDOW times must be accepted once."""
    from config import STABILITY_WINDOW
    p = GestureProcessor()
    results = feed_n(p, "hello", 0.95, STABILITY_WINDOW)
    assert results == ["hello"], f"Expected ['hello'], got {results}"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Repeated identical gestures — only accepted once (dedup + cooldown)
# ──────────────────────────────────────────────────────────────────────────────

def test_repeated_gesture_not_duplicated():
    """Feeding the same gesture many times must never produce duplicates."""
    from config import STABILITY_WINDOW
    p = GestureProcessor()
    results = feed_n(p, "hello", 0.95, STABILITY_WINDOW * 5)
    assert results.count("hello") == 1, (
        f"Expected exactly 1 'hello', got {results.count('hello')}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3. Low confidence — rejected
# ──────────────────────────────────────────────────────────────────────────────

def test_low_confidence_rejected():
    """Predictions below CONFIDENCE_THRESHOLD must always be rejected."""
    from config import CONFIDENCE_THRESHOLD, STABILITY_WINDOW
    p = GestureProcessor()
    low_conf = max(0.0, CONFIDENCE_THRESHOLD - 0.1)
    results = feed_n(p, "water", low_conf, STABILITY_WINDOW * 2)
    assert results == [], f"Expected no accepted gestures, got {results}"


# ──────────────────────────────────────────────────────────────────────────────
# 4. Multiple different gestures — each accepted once in order
# ──────────────────────────────────────────────────────────────────────────────

def test_multiple_gestures_in_order():
    """Distinct gestures must each be accepted exactly once, in order."""
    from config import STABILITY_WINDOW, GESTURE_COOLDOWN
    p = GestureProcessor()
    accepted = []

    gestures = ["hello", "i", "need", "water"]

    # Patch monotonic so cooldown is never a barrier between different words
    fake_time = [0.0]

    def mock_monotonic():
        return fake_time[0]

    with patch("backend.gesture_processor.time.monotonic", side_effect=mock_monotonic):
        for gesture in gestures:
            # Advance time to satisfy cooldown
            fake_time[0] += GESTURE_COOLDOWN + 1.0
            for _ in range(STABILITY_WINDOW):
                result = p.process(gesture, 0.92)
                if result is not None:
                    accepted.append(result)

    assert accepted == gestures, f"Expected {gestures}, got {accepted}"


# ──────────────────────────────────────────────────────────────────────────────
# 5. Cooldown enforced between same gesture
# ──────────────────────────────────────────────────────────────────────────────

def test_cooldown_enforced():
    """Same gesture cannot be accepted twice within GESTURE_COOLDOWN seconds."""
    from config import STABILITY_WINDOW, GESTURE_COOLDOWN
    p = GestureProcessor()

    fake_time = [0.0]

    def mock_monotonic():
        return fake_time[0]

    with patch("backend.gesture_processor.time.monotonic", side_effect=mock_monotonic):
        # First acceptance
        fake_time[0] = GESTURE_COOLDOWN + 1.0
        for _ in range(STABILITY_WINDOW):
            p.process("hello", 0.95)

        # Immediately try again (within cooldown) — should be suppressed
        results_during_cooldown = []
        fake_time[0] += 0.1   # only 0.1s elapsed
        for _ in range(STABILITY_WINDOW):
            r = p.process("hello", 0.95)
            if r is not None:
                results_during_cooldown.append(r)

        assert results_during_cooldown == [], (
            "Gesture accepted during cooldown — expected suppression."
        )

        # After cooldown — should be accepted again
        fake_time[0] += GESTURE_COOLDOWN + 1.0
        # But dedup will block same-as-last-accepted, so use a different one
        results_after = []
        for _ in range(STABILITY_WINDOW):
            r = p.process("world", 0.95)
            if r is not None:
                results_after.append(r)

        assert results_after == ["world"], f"Expected ['world'], got {results_after}"


# ──────────────────────────────────────────────────────────────────────────────
# 6. Mixed high/low confidence in window — only high-conf stabilises
# ──────────────────────────────────────────────────────────────────────────────

def test_mixed_confidence_resets_window():
    """A low-confidence frame in the middle of a window resets accumulation."""
    from config import STABILITY_WINDOW, CONFIDENCE_THRESHOLD
    p = GestureProcessor()

    # Feed STABILITY_WINDOW - 1 high-conf frames, then one low-conf
    for _ in range(STABILITY_WINDOW - 1):
        r = p.process("hello", 0.95)
        assert r is None

    # Low-conf frame — clears window
    r = p.process("hello", CONFIDENCE_THRESHOLD - 0.1)
    assert r is None

    # Now need another full window to accept
    results = feed_n(p, "hello", 0.95, STABILITY_WINDOW)
    assert results == ["hello"]


# ──────────────────────────────────────────────────────────────────────────────
# 7. Reset clears all state
# ──────────────────────────────────────────────────────────────────────────────

def test_reset_clears_state():
    """After reset(), the processor should accept gestures fresh."""
    from config import STABILITY_WINDOW
    p = GestureProcessor()

    # Accept one gesture
    feed_n(p, "hello", 0.95, STABILITY_WINDOW)
    assert p._last_accepted_gesture == "hello"

    p.reset()
    assert p._last_accepted_gesture == ""
    assert len(p._window) == 0


# ──────────────────────────────────────────────────────────────────────────────
# 8. Stability window not met — rejected
# ──────────────────────────────────────────────────────────────────────────────

def test_stability_window_not_met():
    """Fewer than STABILITY_WINDOW frames must never produce an accepted gesture."""
    from config import STABILITY_WINDOW
    p = GestureProcessor()
    results = feed_n(p, "hello", 0.95, STABILITY_WINDOW - 1)
    assert results == [], f"Expected nothing before window filled, got {results}"


# ──────────────────────────────────────────────────────────────────────────────
# 9. Inconsistent gestures in window — rejected
# ──────────────────────────────────────────────────────────────────────────────

def test_inconsistent_window_rejected():
    """If consecutive frames differ, no gesture is accepted."""
    p = GestureProcessor()
    alternating = ["hello", "world", "hello", "world", "hello"]
    results = []
    for g in alternating:
        r = p.process(g, 0.95)
        if r is not None:
            results.append(r)
    assert results == [], f"Expected nothing from inconsistent window, got {results}"
