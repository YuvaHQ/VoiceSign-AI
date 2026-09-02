"""
src/recognition/mock_recognizer.py
----------------------------------
Mock implementation of SignRecognizerInterface for testing.
"""

from typing import List, Optional, Union
import numpy as np

from src.landmarks.sequence import compute_motion_energy
from src.models.database import list_custom_signs
from src.models.schemas import (
    RecognitionResult,
    SampleTypeEnum,
    SignLanguageEnum,
)
from src.recognition.interface import SignRecognizerInterface


class MockSignRecognizer(SignRecognizerInterface):
    def __init__(self, default_confidence: float = 0.92):
        self.default_confidence = default_confidence

    def recognize_sign(
        self,
        input_sequence: Union[List[float], List[List[float]], np.ndarray],
        language: SignLanguageEnum,
    ) -> RecognitionResult:
        seq = np.asarray(input_sequence, dtype=np.float32)
        motion = compute_motion_energy(seq)
        is_dynamic = motion >= 0.038

        if language == SignLanguageEnum.ASL:
            label = "Hello" if is_dynamic else "Book"
        elif language == SignLanguageEnum.ISL:
            label = "Namaste" if not is_dynamic else "Help"
        elif language == SignLanguageEnum.BSL:
            label = "Good Morning" if is_dynamic else "Yes"
        else:
            label = "Custom Sign"

        return RecognitionResult(
            label=label,
            language=language.value,
            confidence=self.default_confidence,
            sample_type=SampleTypeEnum.DYNAMIC if is_dynamic else SampleTypeEnum.STATIC,
            is_custom=False,
            motion_energy=motion,
            is_fallback=True,
            description="Mock Recognizer (ML Model Placeholder)",
        )

    def recognize_custom_sign(
        self,
        user_id: str,
        input_sequence: Union[List[float], List[List[float]], np.ndarray],
    ) -> Optional[RecognitionResult]:
        custom_signs = list_custom_signs(user_id=user_id)
        if not custom_signs:
            return None

        seq = np.asarray(input_sequence, dtype=np.float32)
        motion = compute_motion_energy(seq)
        is_dynamic = motion >= 0.038
        input_flat = seq.flatten()

        best_sign = None
        best_dist = float("inf")

        for sign in custom_signs:
            for sample in sign.samples:
                if sample.features and len(input_flat) >= 126:
                    target_flat = np.asarray(sample.features, dtype=np.float32)
                    dist = float(np.linalg.norm(input_flat[:126] - target_flat))
                    if dist < best_dist:
                        best_dist = dist
                        best_sign = sign
                elif sample.frames and len(input_flat) >= 3780:
                    target_flat = np.asarray(sample.frames, dtype=np.float32).flatten()
                    min_len = min(len(input_flat), len(target_flat))
                    dist = float(np.linalg.norm(input_flat[:min_len] - target_flat[:min_len]))
                    if dist < best_dist:
                        best_dist = dist
                        best_sign = sign

        if best_sign:
            confidence = max(0.5, min(0.98, 1.0 - (best_dist / 10.0)))
            return RecognitionResult(
                label=best_sign.label,
                language="CUSTOM",
                confidence=round(confidence, 2),
                sample_type=SampleTypeEnum.DYNAMIC if is_dynamic else SampleTypeEnum.STATIC,
                is_custom=True,
                motion_energy=motion,
                is_fallback=False,
                description=f"Matched custom sign: {best_sign.label}",
            )

        return None