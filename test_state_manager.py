import pytest
from backend.state_manager import AppState, SPEECH_IDLE, SPEECH_GENERATING, SPEECH_PLAYING, SPEECH_ERROR

def test_app_state_defaults():
    state = AppState()
    assert state.current_gesture == ''
    assert state.gesture_confidence == 0.0
    assert state.raw_sentence == ''
    assert state.ai_enhanced_sentence == ''
    assert state.translation_active is False
    assert state.speech_status == SPEECH_IDLE
    assert state.last_error is None

def test_app_state_update_and_to_dict():
    state = AppState()
    state.update(
        current_gesture='hello',
        gesture_confidence=0.92345,
        raw_sentence='hello',
        translation_active=True,
    )
    d = state.to_dict()
    assert d['current_gesture'] == 'hello'
    assert abs(d['gesture_confidence'] - 0.9235) <= 1e-4
    assert d['raw_sentence'] == 'hello'
    assert d['translation_active'] is True
    assert '_lock' not in d

def test_app_state_error_handling():
    state = AppState()
    state.set_error('Camera failed')
    assert state.last_error == 'Camera failed'
    state.clear_error()
    assert state.last_error is None

def test_app_state_reset():
    state = AppState()
    state.update(
        current_gesture='help',
        gesture_confidence=0.88,
        raw_sentence='i need help',
        ai_enhanced_sentence='I need help.',
        translation_active=True,
        speech_status=SPEECH_PLAYING,
        last_error='Some warning',
    )
    state.reset()
    assert state.current_gesture == ''
    assert state.gesture_confidence == 0.0
    assert state.raw_sentence == ''
    assert state.ai_enhanced_sentence == ''
    assert state.translation_active is False
    assert state.speech_status == SPEECH_IDLE
    assert state.last_error is None
