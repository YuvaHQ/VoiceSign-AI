"""
backend/sentence_builder.py — Sign2Voice
=========================================
Accumulates accepted gestures into a sentence.

Maintains TWO separate values:
  raw_sentence         — exactly what the gestures produced
  ai_enhanced_sentence — the OpenAI-improved version (set externally)

These are NEVER mixed. The UI may display either or both.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SentenceBuilder:
    """
    Builds and manages the sign-language sentence.

    The raw sentence is the ground truth.
    The AI sentence is an optional enhancement layer.
    """

    def __init__(self) -> None:
        self._words: list[str] = []
        self._ai_enhanced_sentence: str = ""
        self._finalized_sentences: list[str] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Word management
    # ──────────────────────────────────────────────────────────────────────────

    def add_word(self, word: str) -> None:
        """Append a word to the raw sentence."""
        word = word.strip().lower()
        if not word:
            logger.debug("add_word: empty word ignored.")
            return
        self._words.append(word)
        # Invalidate AI sentence when raw sentence changes
        self._ai_enhanced_sentence = ""
        logger.info("Word added: '%s' → sentence: '%s'", word, self.get_current_sentence())

    def add_gesture(self, gesture: str) -> None:
        """Alias for add_word — semantically clearer when called from pipeline."""
        self.add_word(gesture)

    def remove_last_word(self) -> Optional[str]:
        """Remove and return the last word. Returns None if sentence is empty."""
        if not self._words:
            logger.debug("remove_last_word: sentence already empty.")
            return None
        removed = self._words.pop()
        # Invalidate AI sentence
        self._ai_enhanced_sentence = ""
        logger.info("Word removed: '%s' → sentence: '%s'", removed, self.get_current_sentence())
        return removed

    def clear_sentence(self) -> None:
        """Remove all words and reset both sentences."""
        self._words.clear()
        self._ai_enhanced_sentence = ""
        logger.info("Sentence cleared.")

    # ──────────────────────────────────────────────────────────────────────────
    # Sentence retrieval
    # ──────────────────────────────────────────────────────────────────────────

    def get_current_sentence(self) -> str:
        """Return the current raw sentence as a space-joined string."""
        return " ".join(self._words)

    def get_ai_sentence(self) -> str:
        """Return the AI-enhanced sentence (empty string if not yet generated)."""
        return self._ai_enhanced_sentence

    def get_word_count(self) -> int:
        return len(self._words)

    def is_empty(self) -> bool:
        return len(self._words) == 0

    # ──────────────────────────────────────────────────────────────────────────
    # AI sentence (set externally by sentence_corrector)
    # ──────────────────────────────────────────────────────────────────────────

    def set_ai_sentence(self, text: str) -> None:
        """Store the AI-enhanced sentence. Called by sentence_corrector."""
        self._ai_enhanced_sentence = text.strip()
        logger.info("AI sentence set: '%s'", self._ai_enhanced_sentence)

    # ──────────────────────────────────────────────────────────────────────────
    # Finalization
    # ──────────────────────────────────────────────────────────────────────────

    def finalize_sentence(self) -> str:
        """
        Commit the current raw sentence to history and reset the buffer.

        Returns:
            The finalized raw sentence string.
        """
        sentence = self.get_current_sentence()
        if sentence:
            self._finalized_sentences.append(sentence)
            logger.info("Sentence finalized: '%s'", sentence)
        self._words.clear()
        self._ai_enhanced_sentence = ""
        return sentence

    def get_history(self) -> list[str]:
        """Return all previously finalized sentences."""
        return list(self._finalized_sentences)

    # ──────────────────────────────────────────────────────────────────────────
    # Full reset (session reset)
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear everything including history."""
        self._words.clear()
        self._ai_enhanced_sentence = ""
        self._finalized_sentences.clear()
        logger.debug("SentenceBuilder fully reset.")
