"""
src/custom_signs/manager.py
---------------------------
Core management logic for custom user-taught signs ("Teach My Sign").
"""
from typing import List, Optional
import uuid

from src.config import MIN_CUSTOM_SAMPLES_REQUIRED
from src.landmarks.normalizer import normalize_landmarks
from src.landmarks.sequence import compute_motion_energy, resample_sequence
from src.models.database import (
    add_samples_to_custom_sign,
    create_custom_sign,
    delete_custom_sign,
    get_custom_sign,
    list_custom_signs,
    update_custom_sign,
)
from src.models.schemas import (
    CreateCustomSignRequest,
    CustomSign,
    CustomSignSampleInput,
    SampleTypeEnum,
    UpdateCustomSignRequest,
)


class CustomSignManager:
    def __init__(self, min_samples: int = MIN_CUSTOM_SAMPLES_REQUIRED):
        self.min_samples = min_samples

    def create_sign(self, request: CreateCustomSignRequest) -> CustomSign:
        if not request.label or not request.label.strip():
            raise ValueError('Label cannot be empty')
        if len(request.samples) < self.min_samples:
            raise ValueError(f'At least {self.min_samples} sample recordings are required')

        sign_id = f"sign_{uuid.uuid4().hex[:8]}"
        processed_samples = []
        for s in request.samples:
            processed_samples.append(self._process_sample_input(s))

        return create_custom_sign(
            sign_id=sign_id,
            user_id=request.user_id,
            label=request.label,
            description=request.description or '',
            samples=processed_samples,
        )

    def _process_sample_input(self, sample_in: CustomSignSampleInput) -> CustomSignSampleInput:
        if sample_in.frames and len(sample_in.frames) > 1:
            resampled = resample_sequence(sample_in.frames, target_length=30)
            norm_frames = [normalize_landmarks(f).tolist() for f in resampled]
            motion = compute_motion_energy(resampled)
            return CustomSignSampleInput(
                sample_type=SampleTypeEnum.DYNAMIC,
                frames=norm_frames,
                motion_energy=motion,
                created_at=sample_in.created_at,
            )
        elif sample_in.features:
            norm_feats = normalize_landmarks(sample_in.features).tolist()
            return CustomSignSampleInput(
                sample_type=SampleTypeEnum.STATIC,
                features=norm_feats,
                motion_energy=0.0,
                created_at=sample_in.created_at,
            )
        elif sample_in.frames and len(sample_in.frames) == 1:
            norm_feats = normalize_landmarks(sample_in.frames[0]).tolist()
            return CustomSignSampleInput(
                sample_type=SampleTypeEnum.STATIC,
                features=norm_feats,
                motion_energy=0.0,
                created_at=sample_in.created_at,
            )
        else:
            raise ValueError("Sample must contain either 'features' (126 floats) or 'frames' (sequence of 126 floats)")

    def list_signs(self, user_id: Optional[str] = None) -> List[CustomSign]:
        return list_custom_signs(user_id=user_id)

    def get_sign(self, sign_id: str) -> Optional[CustomSign]:
        return get_custom_sign(sign_id)

    def update_sign(self, sign_id: str, request: UpdateCustomSignRequest) -> Optional[CustomSign]:
        return update_custom_sign(sign_id, label=request.label, description=request.description)

    def add_samples(self, sign_id: str, new_samples: List[CustomSignSampleInput]) -> Optional[CustomSign]:
        if not new_samples:
            raise ValueError('No new samples provided')
        processed = [self._process_sample_input(s) for s in new_samples]
        return add_samples_to_custom_sign(sign_id, processed)

    def delete_sign(self, sign_id: str) -> bool:
        return delete_custom_sign(sign_id)


_GLOBAL_CUSTOM_SIGN_MANAGER: Optional[CustomSignManager] = None


def get_custom_sign_manager() -> CustomSignManager:
    global _GLOBAL_CUSTOM_SIGN_MANAGER
    if _GLOBAL_CUSTOM_SIGN_MANAGER is None:
        _GLOBAL_CUSTOM_SIGN_MANAGER = CustomSignManager()
    return _GLOBAL_CUSTOM_SIGN_MANAGER
