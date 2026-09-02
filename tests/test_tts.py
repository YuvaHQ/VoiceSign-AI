import pytest
from unittest.mock import patch, MagicMock
from speech.tts import TTSEngine

def test_tts_initialization():
    engine = TTSEngine()
    assert engine.is_available is True
    assert engine.last_spoken_text == ''

def test_tts_synthesize_empty_returns_none():
    engine = TTSEngine()
    assert engine.synthesize('') is None
    assert engine.synthesize('   ') is None

def test_tts_get_autoplay_html():
    engine = TTSEngine()
    html = engine.get_autoplay_html(b'fake_mp3_data')
    assert '<audio autoplay' in html
    assert 'data:audio/mp3;base64,' in html

def test_tts_synthesize_success_mock():
    engine = TTSEngine()
    with patch('gtts.gTTS.write_to_fp') as mock_write:
        def side_effect(buffer):
            buffer.write(b'valid_mp3_audio')
        mock_write.side_effect = side_effect
        res = engine.synthesize('hello')
        assert res == b'valid_mp3_audio'
        assert engine.last_spoken_text == 'hello'
