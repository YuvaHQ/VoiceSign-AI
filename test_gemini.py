"""
tests/test_gemini.py
--------------------
Automated Python tests for the Gemini AI supporting service and fallback behaviors.
"""

import pytest
from unittest.mock import MagicMock
from src.services.gemini_service import GeminiService


def test_gemini_service_missing_key_fallback():
    service = GeminiService(api_key="")
    assert not service.is_available()

    words = ["hello", "friend", "need", "help"]
    sentence, is_gemini, latency = service.polish_sentence(words)
    assert is_gemini is False
    assert sentence == "Hello friend need help."
    assert latency >= 0.0

    desc, is_gem_desc = service.describe_sign("Peace", "ASL")
    assert is_gem_desc is False
    assert "Peace" in desc


def test_gemini_service_mocked_success():
    service = GeminiService(api_key="mock_key_test")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Hello my friend, I need some help."
    mock_client.models.generate_content.return_value = mock_response

    service._client = mock_client

    words = ["HELLO", "FRIEND", "NEED", "HELP"]
    sentence, is_gemini, latency = service.polish_sentence(words)
    assert is_gemini is True
    assert sentence == "Hello my friend, I need some help."


def test_gemini_service_error_handling():
    service = GeminiService(api_key="mock_key_test")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("API Timeout")
    service._client = mock_client

    words = ["thank", "you"]
    sentence, is_gemini, latency = service.polish_sentence(words)
    assert is_gemini is False
    assert sentence == "Thank you."