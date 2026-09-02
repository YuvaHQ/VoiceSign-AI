"""
src/landmarks/sequence.py
-------------------------
Sequence buffering, resampling, motion energy calculation, and sample creation.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Union
import numpy as np

from src.config import SEQUENCE_LENGTH, TOTAL_FRAME_FEATURES
from src.landmarks.normalizer import normalize_landmarks
from src.models.schemas import DynamicSignSample, SampleMetadata, SampleTypeEnum, SignLanguageEnum, StaticSignSample


class GestureSequenceBuffer:
    def __init__(self, sequence_length: int = SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=sequence_length)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, landmarks_126: Union[List[float], np.ndarray]) -> None:
        arr = np.asarray(landmarks_126, dtype=np.float32).flatten()
        if arr.shape[0] != TOTAL_FRAME_FEATURES:
            raise ValueError(f"Expected {TOTAL_FRAME_FEATURES} values, got {arr.shape[0]}.")
        self.buffer.append(arr)

    def is_ready(self) -> bool:
        return len(self.buffer) == self.sequence_length

    def get_latest_frame(self) -> np.ndarray:
        if not self.buffer:
            return np.zeros(TOTAL_FRAME_FEATURES, dtype=np.float32)
        return self.buffer[-1]

    def get_flattened(self) -> np.ndarray:
        if not self.is_ready():
            raise ValueError(f"Buffer not full ({len(self.buffer)}/{self.sequence_length})")
        return np.concatenate(list(self.buffer))

    def get_sequence_matrix(self) -> np.ndarray:
        if not self.buffer:
            return np.zeros((0, TOTAL_FRAME_FEATURES), dtype=np.float32)
        return np.array(list(self.buffer), dtype=np.float32)

    def compute_motion_energy(self) -> float:
        if len(self.buffer) < 5:
            return 0.0
        return compute_motion_energy(self.get_sequence_matrix())

    def clear(self) -> None:
        self.buffer.clear()


def compute_motion_energy(sequence_matrix: Union[List[List[float]], np.ndarray]) -> float:
    seq = np.asarray(sequence_matrix, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[0] < 2 or seq.shape[1] != TOTAL_FRAME_FEATURES:
        return 0.0

    frames_4d = seq.reshape(seq.shape[0], 2, 21, 3)
    lh_wrist_var = np.std(frames_4d[:, 0, 0, :], axis=0) if np.any(frames_4d[:, 0, 0, :] != 0) else [0]
    rh_wrist_var = np.std(frames_4d[:, 1, 0, :], axis=0) if np.any(frames_4d[:, 1, 0, :] != 0) else [0]
    lh_idx_var = np.std(frames_4d[:, 0, 8, :], axis=0) if np.any(frames_4d[:, 0, 8, :] != 0) else [0]
    rh_idx_var = np.std(frames_4d[:, 1, 8, :], axis=0) if np.any(frames_4d[:, 1, 8, :] != 0) else [0]

    total_var = float(np.mean(np.concatenate([lh_wrist_var, rh_wrist_var, lh_idx_var, rh_idx_var])))
    return round(total_var, 5)


def resample_sequence(sequence: Union[List[List[float]], np.ndarray], target_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 2:
        raise ValueError("Sequence must be a 2D array [num_frames, 126]")

    orig_length, num_features = seq.shape
    if orig_length == target_length:
        return seq.copy()

    if orig_length == 1:
        return np.repeat(seq, target_length, axis=0)

    orig_indices = np.linspace(0, orig_length - 1, num=orig_length)
    target_indices = np.linspace(0, orig_length - 1, num=target_length)

    resampled = np.zeros((target_length, num_features), dtype=np.float32)
    for feat_idx in range(num_features):
        resampled[:, feat_idx] = np.interp(target_indices, orig_indices, seq[:, feat_idx])

    return resampled.astype(np.float32)


def create_static_sample(
    features_126: Union[List[float], np.ndarray],
    label: str,
    language: SignLanguageEnum,
    detected_hands: Optional[List[str]] = None,
) -> StaticSignSample:
    norm_feats = normalize_landmarks(features_126)
    return StaticSignSample(
        language=language,
        label=label,
        features=norm_feats.tolist(),
        metadata=SampleMetadata(
            detected_hands=detected_hands or [],
            motion_energy=0.0,
        ),
    )


def create_dynamic_sample(
    frames_sequence: Union[List[List[float]], np.ndarray],
    label: str,
    language: SignLanguageEnum,
    detected_hands: Optional[List[str]] = None,
) -> DynamicSignSample:
    resampled = resample_sequence(frames_sequence, target_length=SEQUENCE_LENGTH)
    norm_frames = [normalize_landmarks(frame).tolist() for frame in resampled]
    motion = compute_motion_energy(resampled)
    return DynamicSignSample(
        language=language,
        label=label,
        frames=norm_frames,
        metadata=SampleMetadata(
            num_frames=len(norm_frames),
            motion_energy=motion,
            detected_hands=detected_hands or [],
        ),
    )


create_dynamic_sequence = create_dynamic_sample
create_static_sequence = create_static_sample
