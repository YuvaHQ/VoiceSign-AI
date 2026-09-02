"""
src/custom_signs/service.py
---------------------------
High-level service layer for custom sign recording sessions,
formatting, and feature aggregation.
"""

from typing import Any, Dict, List, Optional
import time

from src.custom_signs.manager import get_custom_sign_manager
from src.landmarks.extractor import extract_from_base64
from src.landmarks.sequence import compute_motion_energy, resample_sequence
from src.landmarks.normalizer import normalize_landmarks
from src.models.schemas import (
    CustomSignCreateRequest,
    CustomSignRecord,
    CustomSignSampleInput,
    SampleTypeEnum,
)


class CustomSignService:
    """
    Service coordinating custom sign creation from live client sessions.
    """

    def __init__(self):
        self.manager = get_custom_sign_manager()

    def process_and_create(
        self,
        user_id: str,
        label: str,
        description: str,
        raw_samples: List[Dict[str, Any]],
    ) -> CustomSignRecord:
        converted_samples: List[CustomSignSampleInput] = []

        for item in raw_samples:
            sample_type = item.get("sample_type", "dynamic")
            features = item.get("features")
            frames = item.get("frames")
            b64_frames = item.get("b64_frames")

            if b64_frames:
                extracted_frames = []
                for b64 in b64_frames:
                    lms_126, _ = extract_from_base64(b64)
                    extracted_frames.append(lms_126.tolist())

                if len(extracted_frames) == 1:
                    converted_samples.append(
                        CustomSignSampleInput(
                            sample_type=SampleTypeEnum.STATIC,
                            features=extracted_frames[0],
                            motion_energy=0.0,
                            created_at=time.time(),
                        )
                    )
                elif len(extracted_frames) > 1:
                    resampled = resample_sequence(extracted_frames)
                    motion = compute_motion_energy(resampled)
                    converted_samples.append(
                        CustomSignSampleInput(
                            sample_type=SampleTypeEnum.DYNAMIC,
                            frames=[f.tolist() for f in resampled],
                            motion_energy=motion,
                            created_at=time.time(),
                        )
                    )
            elif frames:
                converted_samples.append(
                    CustomSignSampleInput(
                        sample_type=SampleTypeEnum(sample_type),
                        frames=frames,
                        created_at=time.time(),
                    )
                )
            elif features:
                converted_samples.append(
                    CustomSignSampleInput(
                        sample_type=SampleTypeEnum.STATIC,
                        features=features,
                        created_at=time.time(),
                    )
                )

        req = CustomSignCreateRequest(
            user_id=user_id,
            label=label,
            description=description,
            samples=converted_samples,
        )
        return self.manager.create_sign(req)


_GLOBAL_SERVICE: Optional[CustomSignService] = None


def get_custom_sign_service() -> CustomSignService:
    global _GLOBAL_SERVICE
    if _GLOBAL_SERVICE is None:
        _GLOBAL_SERVICE = CustomSignService()
    return _GLOBAL_SERVICE