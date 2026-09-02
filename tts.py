"""
speech/tts.py — Sign2Voice
============================
Text-to-Speech using gTTS (Google Text-to-Speech).

How it works in a Streamlit web app:
  1. gTTS converts text → MP3 bytes (via Google's free TTS API).
  2. The MP3 is returned as bytes so Streamlit can play it with st.audio().
  3. A Streamlit-compatible HTML autoplay component is also provided.

CONTRACT:
  - NEVER called per video frame.
  - ONLY triggered by explicit user action or sentence finalization.
  - Handles empty text, network errors, and API failures gracefully.
  - Always returns bytes or None — never crashes the application.
"""

import io
import logging
from typing import Optional

from config import TTS_LANGUAGE, TTS_SLOW

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Converts text to speech audio bytes using gTTS.

    The caller (Streamlit) is responsible for playing the returned bytes
    via st.audio() or a custom HTML audio component.
    """

    def __init__(self) -> None:
        self._last_text: str = ""
        self._available: bool = self._check_availability()

    def _check_availability(self) -> bool:
        try:
            from gtts import gTTS  # noqa: F401 — just checking import
            logger.info("gTTS TTS engine ready (lang=%s, slow=%s).", TTS_LANGUAGE, TTS_SLOW)
            return True
        except ImportError:
            logger.error(
                "gTTS is not installed. Run: pip install gtts\n"
                "TTS will be disabled."
            )
            return False

    def synthesize(self, text: str) -> Optional[bytes]:
        """
        Convert text to MP3 audio bytes.

        Args:
            text: The sentence to speak.

        Returns:
            MP3 audio as bytes, or None on failure / empty input.
        """
        text = text.strip()

        if not text:
            logger.debug("synthesize: empty text, nothing to speak.")
            return None

        if not self._available:
            logger.warning("synthesize: gTTS not available.")
            return None

        try:
            from gtts import gTTS

            tts = gTTS(text=text, lang=TTS_LANGUAGE, slow=TTS_SLOW)
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            buffer.seek(0)
            audio_bytes = buffer.read()

            self._last_text = text
            logger.info("TTS synthesized %d bytes for: '%s'", len(audio_bytes), text)
            return audio_bytes

        except Exception as exc:
            logger.error("gTTS synthesis failed: %s", exc)
            return None

    def get_autoplay_html(self, audio_bytes: bytes) -> str:
        """
        Return an HTML snippet that autoplays the given MP3 bytes in the browser.
        Use with st.components.v1.html(tts.get_autoplay_html(audio_bytes), height=0).

        Args:
            audio_bytes: MP3 bytes from synthesize().

        Returns:
            HTML string with a hidden <audio autoplay> element.
        """
        import base64
        b64 = base64.b64encode(audio_bytes).decode()
        return (
            f'<audio autoplay style="display:none">'
            f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3">'
            f'</audio>'
        )

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def last_spoken_text(self) -> str:
        return self._last_text
