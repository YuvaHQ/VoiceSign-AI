"""
tests/test_sentence_builder.py — Sign2Voice
============================================
Unit tests for SentenceBuilder.

Covers: add_word, remove_last_word, clear, finalize, raw/AI separation.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from backend.sentence_builder import SentenceBuilder


# ──────────────────────────────────────────────────────────────────────────────
# 1. Empty sentence
# ──────────────────────────────────────────────────────────────────────────────

def test_initially_empty():
    b = SentenceBuilder()
    assert b.get_current_sentence() == ""
    assert b.get_ai_sentence() == ""
    assert b.is_empty()


# ──────────────────────────────────────────────────────────────────────────────
# 2. add_word and get_current_sentence
# ──────────────────────────────────────────────────────────────────────────────

def test_add_words():
    b = SentenceBuilder()
    b.add_word("hello")
    b.add_word("i")
    b.add_word("need")
    b.add_word("water")
    assert b.get_current_sentence() == "hello i need water"
    assert b.get_word_count() == 4


# ──────────────────────────────────────────────────────────────────────────────
# 3. add_gesture is an alias for add_word
# ──────────────────────────────────────────────────────────────────────────────

def test_add_gesture_alias():
    b = SentenceBuilder()
    b.add_gesture("hello")
    b.add_gesture("world")
    assert b.get_current_sentence() == "hello world"


# ──────────────────────────────────────────────────────────────────────────────
# 4. Empty / whitespace words are ignored
# ──────────────────────────────────────────────────────────────────────────────

def test_empty_word_ignored():
    b = SentenceBuilder()
    b.add_word("")
    b.add_word("  ")
    assert b.get_current_sentence() == ""
    assert b.is_empty()


# ──────────────────────────────────────────────────────────────────────────────
# 5. Words are lowercased
# ──────────────────────────────────────────────────────────────────────────────

def test_words_lowercased():
    b = SentenceBuilder()
    b.add_word("HELLO")
    b.add_word("World")
    assert b.get_current_sentence() == "hello world"


# ──────────────────────────────────────────────────────────────────────────────
# 6. remove_last_word
# ──────────────────────────────────────────────────────────────────────────────

def test_remove_last_word():
    b = SentenceBuilder()
    b.add_word("hello")
    b.add_word("world")
    removed = b.remove_last_word()
    assert removed == "world"
    assert b.get_current_sentence() == "hello"


def test_remove_last_word_empty():
    b = SentenceBuilder()
    result = b.remove_last_word()
    assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# 7. Removing last word invalidates AI sentence
# ──────────────────────────────────────────────────────────────────────────────

def test_remove_last_word_clears_ai_sentence():
    b = SentenceBuilder()
    b.add_word("hello")
    b.set_ai_sentence("Hello.")
    assert b.get_ai_sentence() == "Hello."
    b.remove_last_word()
    assert b.get_ai_sentence() == ""


# ──────────────────────────────────────────────────────────────────────────────
# 8. clear_sentence
# ──────────────────────────────────────────────────────────────────────────────

def test_clear_sentence():
    b = SentenceBuilder()
    b.add_word("hello")
    b.add_word("world")
    b.set_ai_sentence("Hello, world.")
    b.clear_sentence()
    assert b.get_current_sentence() == ""
    assert b.get_ai_sentence() == ""
    assert b.is_empty()


# ──────────────────────────────────────────────────────────────────────────────
# 9. finalize_sentence
# ──────────────────────────────────────────────────────────────────────────────

def test_finalize_sentence():
    b = SentenceBuilder()
    b.add_word("hello")
    b.add_word("world")
    finalized = b.finalize_sentence()
    assert finalized == "hello world"
    # Buffer should be cleared
    assert b.get_current_sentence() == ""
    assert b.is_empty()
    # History should contain the sentence
    assert "hello world" in b.get_history()


def test_finalize_empty_sentence():
    b = SentenceBuilder()
    finalized = b.finalize_sentence()
    assert finalized == ""
    assert b.get_history() == []


# ──────────────────────────────────────────────────────────────────────────────
# 10. raw_sentence and ai_enhanced_sentence are kept separate
# ──────────────────────────────────────────────────────────────────────────────

def test_raw_and_ai_are_separate():
    b = SentenceBuilder()
    b.add_word("hello")
    b.add_word("i")
    b.add_word("need")
    b.add_word("water")

    raw = b.get_current_sentence()
    assert raw == "hello i need water"

    b.set_ai_sentence("Hello, I need water.")
    assert b.get_current_sentence() == "hello i need water"   # unchanged
    assert b.get_ai_sentence() == "Hello, I need water."      # separate


# ──────────────────────────────────────────────────────────────────────────────
# 11. Adding a new word invalidates AI sentence
# ──────────────────────────────────────────────────────────────────────────────

def test_new_word_invalidates_ai_sentence():
    b = SentenceBuilder()
    b.add_word("hello")
    b.set_ai_sentence("Hello.")
    b.add_word("world")           # new word → AI sentence stale
    assert b.get_ai_sentence() == ""


# ──────────────────────────────────────────────────────────────────────────────
# 12. reset clears everything including history
# ──────────────────────────────────────────────────────────────────────────────

def test_reset_clears_history():
    b = SentenceBuilder()
    b.add_word("hello")
    b.finalize_sentence()
    assert len(b.get_history()) == 1
    b.reset()
    assert b.get_history() == []
    assert b.is_empty()
