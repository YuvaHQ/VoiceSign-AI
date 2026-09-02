"""
tests/test_pipeline.py — Sign2Voice
=====================================
End-to-end pipeline integration tests using mocked ML input.

Covers: full pipeline flow, OpenAI success/failure, TTS failure, repeated gestures,
        cooldown, clearing, undo, finalization, reset, demo mode.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock

from integration.pipeline import Sign2VoicePipeline
from config import STABILITY_WINDOW, GESTURE_COOLDOWN


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_pipeline() -> Sign2VoicePipeline:
    """Create a fresh pipeline for each test."""
    return Sign2VoicePipeline()


_test_clock = [100.0]


def push_stable(pipeline: Sign2VoicePipeline, gesture: str, confidence: float = 0.95):
    """
    Push a gesture enough times to pass the stability window,
    bypassing cooldown by advancing time across calls.
    """
    from unittest.mock import patch as _patch

    state = None

    def mono():
        val = _test_clock[0]
        _test_clock[0] += GESTURE_COOLDOWN + 0.1
        return val

    with _patch("backend.gesture_processor.time.monotonic", side_effect=mono):
        _test_clock[0] += GESTURE_COOLDOWN + 1.0
        for _ in range(STABILITY_WINDOW):
            state = pipeline.push_ml_prediction({"gesture": gesture, "confidence": confidence})
    return state


# ──────────────────────────────────────────────────────────────────────────────
# 1. Single gesture accepted
# ──────────────────────────────────────────────────────────────────────────────

def test_single_gesture():
    p = make_pipeline()
    p.start()
    push_stable(p, "hello")
    assert "hello" in p.get_state().raw_sentence


# ──────────────────────────────────────────────────────────────────────────────
# 2. Repeated gesture — only one word in sentence
# ──────────────────────────────────────────────────────────────────────────────

def test_repeated_gesture_not_duplicated():
    p = make_pipeline()
    p.start()
    for _ in range(3):           # try to push hello 3× in a row
        push_stable(p, "hello")
    words = p.get_state().raw_sentence.split()
    assert words.count("hello") == 1, f"Expected 1 'hello', got: {words}"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Low confidence — rejected
# ──────────────────────────────────────────────────────────────────────────────

def test_low_confidence_rejected():
    from config import CONFIDENCE_THRESHOLD
    p = make_pipeline()
    p.start()
    low = max(0.0, CONFIDENCE_THRESHOLD - 0.1)
    # Low confidence never accumulates, so push many frames
    for _ in range(STABILITY_WINDOW * 3):
        p.push_ml_prediction({"gesture": "water", "confidence": low})
    assert p.get_state().raw_sentence == ""


# ──────────────────────────────────────────────────────────────────────────────
# 4. Multiple gestures → correct sentence (the core demo scenario)
# ──────────────────────────────────────────────────────────────────────────────

def test_full_sentence_sequence():
    """hello hello hello i i need water → 'hello i need water'"""
    p = make_pipeline()
    p.start()
    for word in ["hello", "i", "need", "water"]:
        push_stable(p, word)
    assert p.get_state().raw_sentence == "hello i need water"


# ──────────────────────────────────────────────────────────────────────────────
# 5. Translation not active — gestures ignored
# ──────────────────────────────────────────────────────────────────────────────

def test_gestures_ignored_when_stopped():
    p = make_pipeline()
    # Do NOT call p.start()
    push_stable(p, "hello")
    assert p.get_state().raw_sentence == ""


# ──────────────────────────────────────────────────────────────────────────────
# 6. Clearing sentence
# ──────────────────────────────────────────────────────────────────────────────

def test_clear_sentence():
    p = make_pipeline()
    p.start()
    push_stable(p, "hello")
    push_stable(p, "world")
    p.clear_sentence()
    assert p.get_state().raw_sentence == ""
    assert p.get_state().ai_enhanced_sentence == ""


# ──────────────────────────────────────────────────────────────────────────────
# 7. Remove last word
# ──────────────────────────────────────────────────────────────────────────────

def test_remove_last_word():
    p = make_pipeline()
    p.start()
    push_stable(p, "hello")
    push_stable(p, "world")
    p.remove_last_word()
    assert p.get_state().raw_sentence == "hello"


# ──────────────────────────────────────────────────────────────────────────────
# 8. Finalize sentence
# ──────────────────────────────────────────────────────────────────────────────

def test_finalize_sentence():
    p = make_pipeline()
    p.start()
    push_stable(p, "hello")
    push_stable(p, "world")
    p.finalize_sentence()
    # After finalization, sentence buffer resets
    assert p.get_state().raw_sentence == ""
    # History contains the finalized sentence
    assert "hello world" in p._builder.get_history()


# ──────────────────────────────────────────────────────────────────────────────
# 9. OpenAI success (mocked)
# ──────────────────────────────────────────────────────────────────────────────

def test_openai_improve_success():
    p = make_pipeline()
    p.start()
    push_stable(p, "hello")
    push_stable(p, "i")
    push_stable(p, "need")
    push_stable(p, "water")

    # Mock OpenAI to return a clean sentence
    with patch.object(
        p._corrector, "improve_sentence",
        return_value="Hello, I need water."
    ):
        p.improve_sentence()

    assert p.get_state().ai_enhanced_sentence == "Hello, I need water."
    assert p.get_state().raw_sentence == "hello i need water"  # unchanged


# ──────────────────────────────────────────────────────────────────────────────
# 10. OpenAI failure — raw sentence preserved, no crash
# ──────────────────────────────────────────────────────────────────────────────

def test_openai_failure_fallback():
    p = make_pipeline()
    p.start()
    push_stable(p, "hello")
    push_stable(p, "world")

    # Mock OpenAI to raise an exception internally (corrector returns raw)
    with patch.object(
        p._corrector, "improve_sentence",
        return_value="hello world"   # fallback = input unchanged
    ):
        p.improve_sentence()

    state = p.get_state()
    assert state.raw_sentence == "hello world"
    assert state.ai_enhanced_sentence == "hello world"   # fallback value
    # No crash, app state is intact
    assert state.last_error is None


# ──────────────────────────────────────────────────────────────────────────────
# 11. Missing API key — corrector returns raw text, no crash
# ──────────────────────────────────────────────────────────────────────────────

def test_missing_api_key_fallback():
    """SentenceCorrector with no API key must return input unchanged."""
    import openai as _openai
    from ai.sentence_corrector import SentenceCorrector

    with patch("ai.sentence_corrector.OPENAI_API_KEY", None):
        corrector = SentenceCorrector()
        result = corrector.improve_sentence("hello i need water")
        assert result == "hello i need water"
        assert not corrector.is_available


# ──────────────────────────────────────────────────────────────────────────────
# 12. TTS failure — no crash
# ──────────────────────────────────────────────────────────────────────────────

def test_tts_failure_no_crash():
    p = make_pipeline()
    p.start()
    push_stable(p, "hello")

    # Make TTS synthesize return None (simulating failure)
    with patch.object(p._tts, "synthesize", return_value=None):
        p.speak_sentence()  # must not raise

    state = p.get_state()
    assert state.speech_status == "error"   # error recorded
    # Application state is still intact
    assert state.raw_sentence == "hello"


# ──────────────────────────────────────────────────────────────────────────────
# 13. Full pipeline: hello hello hello i i need water → hello i need water
# ──────────────────────────────────────────────────────────────────────────────

def test_full_pipeline_demo_sequence():
    """
    The headline end-to-end test.
    Simulates: hello(×3), i(×3), need(×3), water(×3)
    Expected raw: 'hello i need water'
    Expected AI:  'Hello, I need water.'  (mocked)
    """
    p = make_pipeline()
    p.start()

    for word in ["hello", "i", "need", "water"]:
        push_stable(p, word)

    raw = p.get_state().raw_sentence
    assert raw == "hello i need water", f"Unexpected raw: '{raw}'"

    # Mock OpenAI for AI step
    with patch.object(
        p._corrector, "improve_sentence",
        return_value="Hello, I need water."
    ):
        p.improve_sentence()

    ai = p.get_state().ai_enhanced_sentence
    assert ai == "Hello, I need water.", f"Unexpected AI sentence: '{ai}'"

    # Mock TTS for speak step
    dummy_audio = b"fake_mp3_bytes"
    with patch.object(p._tts, "synthesize", return_value=dummy_audio):
        p.speak_sentence(use_ai=True)

    audio = p.get_last_audio_bytes()
    assert audio == dummy_audio


# ──────────────────────────────────────────────────────────────────────────────
# 14. Reset session
# ──────────────────────────────────────────────────────────────────────────────

def test_reset_session():
    p = make_pipeline()
    p.start()
    push_stable(p, "hello")
    p.reset_session()
    state = p.get_state()
    assert state.raw_sentence == ""
    assert state.translation_active is False
    assert state.current_gesture == ""


# ──────────────────────────────────────────────────────────────────────────────
# 15. Invalid ML output handled gracefully
# ──────────────────────────────────────────────────────────────────────────────

def test_invalid_ml_output_handled():
    p = make_pipeline()
    p.start()
    # Push completely malformed output
    p.push_ml_prediction({"garbage": "data"})
    p.push_ml_prediction(None)   # type: ignore
    p.push_ml_prediction("not a dict")   # type: ignore
    # No crash, sentence unchanged
    assert p.get_state().raw_sentence == ""


# ──────────────────────────────────────────────────────────────────────────────
# 16. ML adapter — all supported input schemas
# ──────────────────────────────────────────────────────────────────────────────

def test_ml_adapter_all_schemas():
    from integration.ml_adapter import MLAdapter
    adapter = MLAdapter()

    schemas = [
        {"gesture": "hello",    "confidence": 0.94},
        {"label": "hello",      "score": 0.94},
        {"prediction": "hello", "prob": 0.94},
        {"class": "hello",      "confidence": 0.94},
        {"gesture": "hello",    "score": 0.94},
        {"sign": "hello",       "probability": 0.94},
    ]

    for schema in schemas:
        result = adapter.normalize(schema)
        assert result is not None, f"Failed to normalise: {schema}"
        assert result["gesture"] == "hello"
        assert abs(result["confidence"] - 0.94) < 1e-6
