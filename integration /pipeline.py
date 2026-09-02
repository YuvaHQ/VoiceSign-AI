"""
integration/pipeline.py — Sign2Voice
======================================
Master orchestrator: wires all modules into a single callable interface.

This is the ONE entry point for:
  - The Streamlit UI
  - The ML team's gesture classifier

ML team usage:
    pipeline = Sign2VoicePipeline()
    pipeline.start()
    # Every time your model produces a prediction:
    state = pipeline.push_ml_prediction({"gesture": "hello", "confidence": 0.94})

Frontend usage:
    state = pipeline.get_state()            # read current state
    state = pipeline.speak_sentence()       # trigger TTS
    state = pipeline.improve_sentence()     # call OpenAI
    state = pipeline.remove_last_word()     # undo
    state = pipeline.clear_sentence()       # clear all
    state = pipeline.finalize_sentence()    # commit + optionally speak
    state = pipeline.reset_session()        # full reset
"""

import logging
from typing import Optional

from backend.gesture_processor import GestureProcessor
from backend.sentence_builder import SentenceBuilder
from backend.state_manager import AppState, SPEECH_IDLE, SPEECH_GENERATING, SPEECH_PLAYING, SPEECH_ERROR
from ai.sentence_corrector import SentenceCorrector
from speech.tts import TTSEngine
from integration.ml_adapter import MLAdapter

logger = logging.getLogger(__name__)


class Sign2VoicePipeline:
    """
    The central pipeline for Sign2Voice.

    Internal flow on push_ml_prediction():
        raw ML dict
          → MLAdapter.normalize()          (schema normalisation)
          → GestureProcessor.process()     (confidence + stability + cooldown)
          → SentenceBuilder.add_gesture()  (word accumulation)
          → AppState.update()              (state sync)

    OpenAI and TTS are NEVER triggered automatically — always event-driven.
    """

    def __init__(self) -> None:
        self._adapter = MLAdapter()
        self._processor = GestureProcessor()
        self._builder = SentenceBuilder()
        self._corrector = SentenceCorrector()
        self._tts = TTSEngine()
        self._state = AppState()
        self._last_audio_bytes: Optional[bytes] = None
        logger.info("Sign2VoicePipeline initialised.")

    # ──────────────────────────────────────────────────────────────────────────
    # Session control
    # ──────────────────────────────────────────────────────────────────────────

    def start(self) -> AppState:
        """Enable gesture acceptance (translation active)."""
        self._state.update(translation_active=True, last_error=None)
        logger.info("Pipeline started.")
        return self._state

    def stop(self) -> AppState:
        """Pause gesture acceptance (keeps current sentence intact)."""
        self._state.update(translation_active=False)
        logger.info("Pipeline stopped.")
        return self._state

    def reset_session(self) -> AppState:
        """Full reset: clear sentence, processor state, and app state."""
        self._processor.reset()
        self._builder.reset()
        self._state.reset()
        self._last_audio_bytes = None
        logger.info("Session reset.")
        return self._state

    # ──────────────────────────────────────────────────────────────────────────
    # ML integration — primary entry point for the ML team
    # ──────────────────────────────────────────────────────────────────────────

    def push_ml_prediction(self, raw_ml_output: dict) -> AppState:
        """
        Feed one ML prediction frame into the pipeline.

        The ML team calls this on every frame where their model produces output.

        Args:
            raw_ml_output: Any supported dict format, e.g.:
                {"gesture": "hello", "confidence": 0.94}
                {"label": "hello",   "score": 0.94}

        Returns:
            Current AppState after processing.
        """
        # ── Guard: only process when translation is active ────────────────────
        if not self._state.translation_active:
            return self._state

        # ── 1. Normalise ML output ────────────────────────────────────────────
        normalised = self._adapter.normalize(raw_ml_output)
        if normalised is None:
            self._state.set_error("Invalid ML output format — see logs.")
            return self._state

        gesture = normalised["gesture"]
        confidence = normalised["confidence"]

        # Update live display (even if not yet accepted)
        self._state.update(
            current_gesture=gesture,
            gesture_confidence=confidence,
            last_error=None,
        )

        # ── 2. Gesture processing (confidence + stability + cooldown) ─────────
        accepted = self._processor.process(gesture, confidence)
        if accepted is None:
            return self._state  # Not stable yet — update display only

        # ── 3. Add to sentence ────────────────────────────────────────────────
        self._builder.add_gesture(accepted)
        self._state.update(
            raw_sentence=self._builder.get_current_sentence(),
            ai_enhanced_sentence="",   # invalidated by new word
        )

        logger.info(
            "Gesture '%s' added → sentence: '%s'",
            accepted, self._state.raw_sentence,
        )
        return self._state

    # ──────────────────────────────────────────────────────────────────────────
    # Sentence operations
    # ──────────────────────────────────────────────────────────────────────────

    def remove_last_word(self) -> AppState:
        """Remove the last word from the current sentence."""
        removed = self._builder.remove_last_word()
        current = self._builder.get_current_sentence()
        self._processor._last_accepted_gesture = current.split()[-1] if current else ""
        self._state.update(
            raw_sentence=current,
            ai_enhanced_sentence=self._builder.get_ai_sentence(),
        )
        if removed:
            logger.info("Removed last word: '%s'", removed)
        return self._state

    def clear_sentence(self) -> AppState:
        """Clear the entire sentence and reset gesture processor."""
        self._builder.clear_sentence()
        self._processor.reset()
        self._state.update(
            raw_sentence="",
            ai_enhanced_sentence="",
            current_gesture="",
            gesture_confidence=0.0,
        )
        logger.info("Sentence cleared.")
        return self._state

    def finalize_sentence(self, auto_speak: bool = False) -> AppState:
        """
        Commit the current sentence to history and reset for the next sentence.
        Optionally trigger TTS after finalization.

        Args:
            auto_speak: If True, synthesize TTS using the AI sentence (if available)
                        or the raw sentence.
        Returns:
            Current AppState.
        """
        raw = self._builder.finalize_sentence()
        self._processor.reset()
        self._state.update(
            raw_sentence="",
            ai_enhanced_sentence="",
            current_gesture="",
            gesture_confidence=0.0,
        )
        logger.info("Sentence finalized: '%s'", raw)

        if auto_speak and raw:
            self._synthesize(raw)

        return self._state

    # ──────────────────────────────────────────────────────────────────────────
    # OpenAI integration (event-driven only)
    # ──────────────────────────────────────────────────────────────────────────

    def improve_sentence(self) -> AppState:
        """
        Request AI improvement of the current raw sentence.
        Only called on explicit user action — NEVER per frame.

        Returns:
            Updated AppState with ai_enhanced_sentence populated.
        """
        raw = self._builder.get_current_sentence()
        if not raw:
            logger.info("improve_sentence: nothing to improve (empty sentence).")
            return self._state

        self._state.update(speech_status=SPEECH_GENERATING)
        improved = self._corrector.improve_sentence(raw)
        self._builder.set_ai_sentence(improved)
        self._state.update(
            ai_enhanced_sentence=improved,
            speech_status=SPEECH_IDLE,
        )
        return self._state

    # ──────────────────────────────────────────────────────────────────────────
    # TTS (event-driven only)
    # ──────────────────────────────────────────────────────────────────────────

    def speak_sentence(self, use_ai: bool = True) -> AppState:
        """
        Synthesize TTS for the current sentence.
        Only called on explicit user action or finalization — NEVER per frame.

        Args:
            use_ai: If True and an AI-enhanced sentence exists, speak that.
                    Otherwise speak the raw sentence.

        Returns:
            Updated AppState. Retrieve audio via get_last_audio_bytes().
        """
        ai_sentence = self._builder.get_ai_sentence()
        raw_sentence = self._builder.get_current_sentence()

        text = (ai_sentence if use_ai and ai_sentence else raw_sentence)
        if not text:
            logger.info("speak_sentence: nothing to speak.")
            return self._state

        self._synthesize(text)
        return self._state

    def _synthesize(self, text: str) -> None:
        """Internal TTS synthesis. Updates state and stores audio bytes."""
        self._state.update(speech_status=SPEECH_GENERATING)
        audio_bytes = self._tts.synthesize(text)

        if audio_bytes:
            self._last_audio_bytes = audio_bytes
            self._state.update(speech_status=SPEECH_PLAYING)
            logger.info("TTS audio ready (%d bytes).", len(audio_bytes))
        else:
            self._last_audio_bytes = None
            self._state.update(
                speech_status=SPEECH_ERROR,
                last_error="TTS synthesis failed. Check internet connection.",
            )

    def get_last_audio_bytes(self) -> Optional[bytes]:
        """
        Return the most recently synthesized audio bytes, then clear them.
        Call after speak_sentence() to retrieve audio for st.audio().
        """
        audio = self._last_audio_bytes
        self._last_audio_bytes = None
        # Reset status to idle after retrieval
        if self._state.speech_status == SPEECH_PLAYING:
            self._state.update(speech_status=SPEECH_IDLE)
        return audio

    # ──────────────────────────────────────────────────────────────────────────
    # State access
    # ──────────────────────────────────────────────────────────────────────────

    def get_state(self) -> AppState:
        """Return the current AppState object."""
        return self._state

    def get_state_dict(self) -> dict:
        """Return state as a plain dict (JSON-serialisable)."""
        return self._state.to_dict()

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostics / Demo mode
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def openai_available(self) -> bool:
        return self._corrector.is_available

    @property
    def tts_available(self) -> bool:
        return self._tts.is_available

    def run_demo_sequence(self) -> AppState:
        """
        Inject a scripted gesture sequence for demo/testing purposes.
        ONLY used in DEMO_MODE. Clearly isolated from the real pipeline.
        """
        from config import DEMO_GESTURE_SEQUENCE
        self.start()
        for frame in DEMO_GESTURE_SEQUENCE:
            self.push_ml_prediction(frame)
        logger.info("Demo sequence complete. Sentence: '%s'", self._state.raw_sentence)
        return self._state
