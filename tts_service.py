"""
src/services/tts_service.py
---------------------------
Text-to-Speech (TTS) Service bridge for Meeting Mode and live accessibility feedback.
Web frontend utilizes Web Speech API by default, while backend provides server-side metadata.
"""

from typing import Any, Dict, Optional
import time

from src.models.schemas import TTSRequest, TTSResponse


class TTSService:
    """
    TTS coordination service.
    """

    def __init__(self):
        pass

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        """
        Processes TTS request and returns metadata for client-side Web Speech synthesis.
        """
        text = request.text.strip()
        return TTSResponse(
            text=text,
            audio_base64=None,
            synthesized=bool(text),
        )


_GLOBAL_TTS_SERVICE: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    global _GLOBAL_TTS_SERVICE
    if _GLOBAL_TTS_SERVICE is None:
        _GLOBAL_TTS_SERVICE = TTSService()
    return _GLOBAL_TTS_SERVICE