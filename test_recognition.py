"""
tests/test_recognition.py
-------------------------
Automated Python tests for the recognition interface, mock recognizer,
5-second persistent safety help detection, meeting debouncing, and hybrid engine.
"""

import time
import pytest
import numpy as np

from src.landmarks.sequence import GestureSequenceBuffer
from src.models.schemas import (
    EventTypeEnum,
    RecognitionResult,
    SampleTypeEnum,
    SignLanguageEnum,
)
from src.recognition.debouncer import MeetingDebouncer
from src.recognition.help_detector import HelpDetector
from src.recognition.hybrid_engine import HybridRecognitionEngine
from src.recognition.mock_recognizer import MockSignRecognizer


def test_mock_sign_recognizer():
    recognizer = MockSignRecognizer(default_confidence=0.95)

    static_frame = [0.1] * 126
    res_static = recognizer.recognize_sign(static_frame, SignLanguageEnum.ASL)
    assert res_static.label == "Book"
    assert res_static.sample_type == SampleTypeEnum.STATIC
    assert res_static.confidence == 0.95

    dyn_frames = [[0.1 + f * 0.05] * 126 for f in range(30)]
    res_dyn = recognizer.recognize_sign(dyn_frames, SignLanguageEnum.ASL)
    assert res_dyn.label == "Hello"
    assert res_dyn.sample_type == SampleTypeEnum.DYNAMIC


def test_help_detector_five_second_persistence():
    detector = HelpDetector(persistence_seconds=2.0)

    help_res = RecognitionResult(
        label="HELP",
        language="ISL",
        confidence=0.95,
        sample_type=SampleTypeEnum.DYNAMIC,
    )
    other_res = RecognitionResult(
        label="Hello",
        language="ISL",
        confidence=0.90,
        sample_type=SampleTypeEnum.DYNAMIC,
    )

    is_active, dur, event = detector.process(help_res)
    assert is_active is True
    assert event is None

    time.sleep(1.0)
    is_active, dur, event = detector.process(help_res)
    assert is_active is True
    assert event is None

    time.sleep(1.1)
    is_active, dur, event = detector.process(help_res)
    assert is_active is True
    assert event is not None
    assert event.event == EventTypeEnum.HELP_DETECTED
    assert event.help_active is True
    assert dur >= 2.0

    time.sleep(0.7)
    is_active, dur, event = detector.process(other_res)
    assert is_active is False
    assert event is None


def test_meeting_debouncer_and_transcript():
    debouncer = MeetingDebouncer(debounce_interval=1.0)

    res_hello = RecognitionResult(
        label="Hello",
        language="ASL",
        confidence=0.92,
        sample_type=SampleTypeEnum.DYNAMIC,
    )
    res_friend = RecognitionResult(
        label="Friend",
        language="ASL",
        confidence=0.90,
        sample_type=SampleTypeEnum.DYNAMIC,
    )

    appended, event = debouncer.process(res_hello)
    assert appended is True
    assert event is not None
    assert debouncer.get_transcript() == "Hello"

    appended2, event2 = debouncer.process(res_hello)
    assert appended2 is False
    assert event2 is None
    assert debouncer.get_transcript() == "Hello"

    appended3, event3 = debouncer.process(res_friend)
    assert appended3 is True
    assert debouncer.get_transcript() == "Hello Friend"

    debouncer.clear()
    assert debouncer.get_transcript() == ""


def test_hybrid_engine_realtime_buffer():
    engine = HybridRecognitionEngine()
    buf = GestureSequenceBuffer(sequence_length=30)

    for i in range(30):
        buf.push([0.05 * i] * 126)

    res, events = engine.process_sequence_buffer(buf, language=SignLanguageEnum.ASL)
    assert res is not None
    assert res.label != ""
    assert len(events) >= 1
    assert any(e.event == EventTypeEnum.SIGN_RECOGNIZED for e in events)