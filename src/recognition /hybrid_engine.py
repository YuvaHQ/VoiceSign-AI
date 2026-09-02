"""
src/recognition/hybrid_engine.py
--------------------------------
Hybrid Recognition Engine combining trained models, custom signs, Help safety monitor, and debouncer.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np

from src.config import (
    CONFIDENCE_THRESHOLD,
    DYNAMIC_MODEL_PATH,
    MOTION_ENERGY_THRESHOLD,
    STATIC_MODEL_PATH,
    TOTAL_FRAME_FEATURES,
    SEQUENCE_LENGTH,
)
from src.landmarks.sequence import GestureSequenceBuffer, compute_motion_energy
from src.models.schemas import (
    EventTypeEnum,
    RecognitionEvent,
    RecognitionResult,
    SampleTypeEnum,
    SignLanguageEnum,
)
from src.recognition.debouncer import get_global_debouncer
from src.recognition.help_detector import get_global_help_detector
from src.recognition.interface import SignRecognizerInterface
from src.recognition.mock_recognizer import MockSignRecognizer


class HybridRecognitionEngine(SignRecognizerInterface):
    def __init__(
        self,
        static_model_path: Union[str, Path] = STATIC_MODEL_PATH,
        dynamic_model_path: Union[str, Path] = DYNAMIC_MODEL_PATH,
        motion_threshold: float = MOTION_ENERGY_THRESHOLD,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.static_model_path = Path(static_model_path)
        self.dynamic_model_path = Path(dynamic_model_path)
        self.motion_threshold = motion_threshold
        self.confidence_threshold = confidence_threshold

        self._static_bundle: Optional[Dict[str, Any]] = None
        self._dynamic_bundle: Optional[Dict[str, Any]] = None
        self._mock_recognizer = MockSignRecognizer()

        self.help_detector = get_global_help_detector()
        self.debouncer = get_global_debouncer()

        self._load_models_if_available()

    def _load_models_if_available(self) -> None:
        if self.static_model_path.exists():
            try:
                self._static_bundle = joblib.load(self.static_model_path)
            except Exception:
                self._static_bundle = None

        if self.dynamic_model_path.exists():
            try:
                self._dynamic_bundle = joblib.load(self.dynamic_model_path)
            except Exception:
                self._dynamic_bundle = None

    def reload_models(self) -> None:
        self._load_models_if_available()

    def recognize_custom_sign(
        self,
        user_id: str,
        input_sequence: Union[List[float], List[List[float]], np.ndarray],
    ) -> Optional[RecognitionResult]:
        return self._mock_recognizer.recognize_custom_sign(user_id, input_sequence)

    def recognize_sign(
        self,
        input_sequence: Union[List[float], List[List[float]], np.ndarray],
        language: SignLanguageEnum = SignLanguageEnum.ASL,
    ) -> RecognitionResult:
        seq = np.asarray(input_sequence, dtype=np.float32)
        motion_energy = compute_motion_energy(seq)
        is_dynamic = motion_energy >= self.motion_threshold

        if language == SignLanguageEnum.CUSTOM:
            custom_res = self.recognize_custom_sign("default_user", input_sequence)
            if custom_res:
                return custom_res

        if is_dynamic and self._dynamic_bundle is not None:
            try:
                clf = self._dynamic_bundle["model"]
                flat_seq = seq.flatten().reshape(1, -1)
                if flat_seq.shape[1] == SEQUENCE_LENGTH * TOTAL_FRAME_FEATURES:
                    pred = clf.predict(flat_seq)[0]
                    conf = float(np.max(clf.predict_proba(flat_seq)[0])) if hasattr(clf, "predict_proba") else 0.90
                    return RecognitionResult(
                        label=str(pred),
                        language=language.value,
                        confidence=round(conf, 2),
                        sample_type=SampleTypeEnum.DYNAMIC,
                        is_custom=False,
                        motion_energy=motion_energy,
                        is_fallback=False,
                    )
            except Exception:
                pass

        if not is_dynamic and self._static_bundle is not None:
            try:
                clf = self._static_bundle["model"]
                frame_126 = seq[-1] if seq.ndim > 1 else seq[:TOTAL_FRAME_FEATURES]
                frame_126 = frame_126.reshape(1, -1)
                if frame_126.shape[1] == TOTAL_FRAME_FEATURES:
                    pred = clf.predict(frame_126)[0]
                    conf = float(np.max(clf.predict_proba(frame_126)[0])) if hasattr(clf, "predict_proba") else 0.90
                    return RecognitionResult(
                        label=str(pred),
                        language=language.value,
                        confidence=round(conf, 2),
                        sample_type=SampleTypeEnum.STATIC,
                        is_custom=False,
                        motion_energy=motion_energy,
                        is_fallback=False,
                    )
            except Exception:
                pass

        return self._mock_recognizer.recognize_sign(input_sequence, language)

    def process_sequence_buffer(
        self,
        buffer: GestureSequenceBuffer,
        language: SignLanguageEnum = SignLanguageEnum.ASL,
        user_id: str = "default_user",
    ) -> Tuple[RecognitionResult, List[RecognitionEvent]]:
        seq_matrix = buffer.get_sequence_matrix()
        if len(seq_matrix) == 0:
            res = RecognitionResult(
                label="No Hand Detected",
                language=language.value,
                confidence=0.0,
                sample_type=SampleTypeEnum.STATIC,
                is_custom=False,
                motion_energy=0.0,
            )
            return res, []

        res = self.recognize_sign(seq_matrix, language)
        events: List[RecognitionEvent] = []

        sign_event = RecognitionEvent(
            event=EventTypeEnum.SIGN_RECOGNIZED,
            label=res.label,
            language=res.language,
            confidence=res.confidence,
            sample_type=res.sample_type,
            is_custom=res.is_custom,
            motion_energy=res.motion_energy,
        )
        events.append(sign_event)

        is_help, help_dur, help_event = self.help_detector.process(res)
        if help_event is not None:
            events.append(help_event)

        appended, trans_event = self.debouncer.process(res)
        if trans_event is not None:
            events.append(trans_event)

        return res, events


_GLOBAL_HYBRID_ENGINE: Optional[HybridRecognitionEngine] = None


def get_hybrid_engine() -> HybridRecognitionEngine:
    global _GLOBAL_HYBRID_ENGINE
    if _GLOBAL_HYBRID_ENGINE is None:
        _GLOBAL_HYBRID_ENGINE = HybridRecognitionEngine()
    return _GLOBAL_HYBRID_ENGINE
