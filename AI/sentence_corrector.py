"""
ai/sentence_corrector.py — Sign2Voice
=======================================
OpenAI-powered sentence improvement.

CONTRACT:
  - NEVER called per video frame.
  - ONLY called on explicit user action or sentence finalization.
  - ALWAYS returns a string — either improved or the original (on any failure).
  - NEVER crashes the application.
  - NEVER logs the API key.

The ML model recognises gestures. OpenAI improves the resulting text only.
"""

import logging
from typing import Optional

import openai

from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TIMEOUT

logger = logging.getLogger(__name__)

# System prompt — strict, concise, meaning-preserving
_SYSTEM_PROMPT = (
    "You are a grammar and punctuation corrector. "
    "The user will give you a sentence assembled from sign-language gestures. "
    "Fix grammar, capitalisation, and punctuation only. "
    "Preserve the original meaning exactly. "
    "Never add information that was not in the input. "
    "Return ONLY the corrected sentence — no explanation, no prefix."
)


class SentenceCorrector:
    """
    Wraps OpenAI chat completions for sentence improvement.

    Gracefully handles every known failure mode and always returns a string.
    """

    def __init__(self) -> None:
        self._client: Optional[openai.OpenAI] = None
        self._available: bool = False
        self._init_client()

    def _init_client(self) -> None:
        """Attempt to create an OpenAI client. Safe to call even without a key."""
        if not OPENAI_API_KEY:
            logger.warning(
                "OPENAI_API_KEY not set. AI sentence improvement will be disabled. "
                "Raw sentences will be used as fallback."
            )
            self._available = False
            return

        try:
            self._client = openai.OpenAI(
                api_key=OPENAI_API_KEY,
                timeout=OPENAI_TIMEOUT,
            )
            self._available = True
            logger.info("OpenAI client initialised (model: %s).", OPENAI_MODEL)
        except Exception as exc:
            logger.error("Failed to initialise OpenAI client: %s", exc)
            self._available = False

    def improve_sentence(self, text: str) -> str:
        """
        Improve grammar and punctuation using OpenAI.

        Args:
            text: Raw sentence from gesture recognition (e.g. "hello i need water").

        Returns:
            Improved sentence (e.g. "Hello, I need water."), or `text` unchanged
            if OpenAI is unavailable or fails for any reason.
        """
        text = text.strip()

        if not text:
            logger.debug("improve_sentence: empty input, nothing to improve.")
            return text

        if not self._available or self._client is None:
            logger.info(
                "improve_sentence: OpenAI unavailable, returning raw sentence."
            )
            return text

        try:
            response = self._client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=256,
                temperature=0.3,
            )

            improved = response.choices[0].message.content
            if not improved:
                logger.warning("OpenAI returned empty response; using raw sentence.")
                return text

            improved = improved.strip()
            logger.info("Sentence improved: '%s' → '%s'", text, improved)
            return improved

        except openai.AuthenticationError:
            logger.error(
                "OpenAI authentication failed. Check OPENAI_API_KEY in .env. "
                "Returning raw sentence."
            )
            self._available = False   # don't retry on every call
            return text

        except openai.RateLimitError:
            logger.warning(
                "OpenAI rate limit reached. Returning raw sentence."
            )
            return text

        except openai.APITimeoutError:
            logger.warning(
                "OpenAI request timed out (%.1fs). Returning raw sentence.",
                OPENAI_TIMEOUT,
            )
            return text

        except openai.APIConnectionError as exc:
            logger.warning(
                "OpenAI connection error: %s. Returning raw sentence.", exc
            )
            return text

        except openai.APIStatusError as exc:
            logger.warning(
                "OpenAI API error (status %s): %s. Returning raw sentence.",
                exc.status_code, exc.message,
            )
            return text

        except Exception as exc:
            logger.error(
                "Unexpected error during OpenAI call: %s. Returning raw sentence.",
                exc,
            )
            return text

    @property
    def is_available(self) -> bool:
        """True if OpenAI client is configured and ready."""
        return self._available
