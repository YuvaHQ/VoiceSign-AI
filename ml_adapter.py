"""
integration/ml_adapter.py — Sign2Voice
========================================
Normalises raw ML model output into a consistent internal format.

The ML team can use any of several common output field names.
This adapter absorbs the difference so nothing else in the codebase
needs to know the exact ML output schema.

Internal canonical format:
    {"gesture": str, "confidence": float}

Supported input schemas (any of these will be accepted):
    {"gesture": "hello",    "confidence": 0.94}   ← canonical
    {"label": "hello",      "score": 0.94}
    {"prediction": "hello", "prob": 0.94}
    {"class": "hello",      "confidence": 0.94}
    {"gesture": "hello",    "score": 0.94}         ← mixed keys OK
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Ordered lists of field names tried for gesture and confidence
_GESTURE_KEYS = ("gesture", "label", "prediction", "class", "sign", "word")
_CONFIDENCE_KEYS = ("confidence", "score", "prob", "probability", "certainty")


class MLAdapter:
    """
    Normalises arbitrary ML output dicts into the canonical gesture format.

    Usage (ML team):
        adapter = MLAdapter()
        result = adapter.normalize(your_model_output)
        if result:
            pipeline.push_ml_prediction(result)
    """

    def normalize(self, raw: object) -> Optional[dict]:
        """
        Parse and normalise a raw ML prediction.

        Args:
            raw: Any object from the ML model (expected to be a dict).

        Returns:
            {"gesture": str, "confidence": float}, or None if parsing fails.
        """
        if not isinstance(raw, dict):
            logger.warning(
                "MLAdapter.normalize: expected dict, got %s. Ignoring.",
                type(raw).__name__,
            )
            return None

        # ── Extract gesture ───────────────────────────────────────────────────
        gesture: Optional[str] = None
        for key in _GESTURE_KEYS:
            if key in raw and isinstance(raw[key], str):
                gesture = raw[key].strip().lower()
                break

        if not gesture:
            logger.warning(
                "MLAdapter.normalize: could not find gesture field in %s. "
                "Expected one of: %s",
                list(raw.keys()), _GESTURE_KEYS,
            )
            return None

        # ── Extract confidence ────────────────────────────────────────────────
        confidence: Optional[float] = None
        for key in _CONFIDENCE_KEYS:
            if key in raw:
                try:
                    confidence = float(raw[key])
                    break
                except (TypeError, ValueError):
                    continue

        if confidence is None:
            logger.warning(
                "MLAdapter.normalize: could not find confidence field in %s. "
                "Expected one of: %s",
                list(raw.keys()), _CONFIDENCE_KEYS,
            )
            return None

        # ── Clamp confidence to [0, 1] ────────────────────────────────────────
        confidence = max(0.0, min(1.0, confidence))

        normalised = {"gesture": gesture, "confidence": confidence}
        logger.debug("MLAdapter: normalised %s → %s", raw, normalised)
        return normalised
