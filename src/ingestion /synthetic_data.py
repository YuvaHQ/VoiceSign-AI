"""
src/ingestion/synthetic_data.py
-------------------------------
Generates realistic 126D synthetic landmark data for unit testing & CI.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.config import FEATURES_PER_HAND, LANDMARKS_PER_HAND, SEQUENCE_LENGTH, TOTAL_FRAME_FEATURES
from src.ingestion.asl_adapter import ASLDatasetAdapter
from src.ingestion.bsl_adapter import BSLDatasetAdapter
from src.ingestion.isl_adapter import ISLDatasetAdapter
from src.models.schemas import SignLanguageEnum


def _generate_canonical_hand_pose(
    wrist_x: float = 0.5,
    wrist_y: float = 0.5,
    finger_spread: float = 0.05,
    extended_fingers: Tuple[bool, bool, bool, bool, bool] = (True, True, True, True, True),
    noise_std: float = 0.005,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.RandomState()

    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[0] = [wrist_x, wrist_y, 0.0]

    for finger_idx in range(5):
        is_ext = extended_fingers[finger_idx]
        base_angle = (finger_idx - 2) * finger_spread
        base_lm = 1 + finger_idx * 4
        for joint in range(4):
            lm_idx = base_lm + joint
            dist = (joint + 1) * (0.06 if is_ext else 0.02)
            landmarks[lm_idx] = [
                wrist_x + np.sin(base_angle) * dist,
                wrist_y - np.cos(base_angle) * dist if is_ext else wrist_y - 0.02,
                0.01 * joint,
            ]

    if noise_std > 0:
        landmarks += rng.normal(0, noise_std, landmarks.shape).astype(np.float32)

    return landmarks.flatten()


def generate_static_sample_features(
    label: str,
    two_handed: bool = False,
    noise_std: float = 0.005,
    rng: Optional[np.random.RandomState] = None,
) -> List[float]:
    if rng is None:
        rng = np.random.RandomState()

    rh = _generate_canonical_hand_pose(wrist_x=0.6, wrist_y=0.5, noise_std=noise_std, rng=rng)
    if two_handed:
        lh = _generate_canonical_hand_pose(wrist_x=0.4, wrist_y=0.5, noise_std=noise_std, rng=rng)
    else:
        lh = np.zeros(FEATURES_PER_HAND, dtype=np.float32)

    return np.concatenate([lh, rh]).tolist()


def generate_dynamic_sequence_features(
    label: str,
    two_handed: bool = False,
    num_frames: int = SEQUENCE_LENGTH,
    motion_type: str = 'wave',
    noise_std: float = 0.005,
    rng: Optional[np.random.RandomState] = None,
) -> List[List[float]]:
    if rng is None:
        rng = np.random.RandomState()

    frames = []
    for f in range(num_frames):
        t = f / float(max(1, num_frames - 1))
        offset_x = 0.1 * np.sin(t * np.pi * 2) if motion_type == 'wave' else 0.15 * t
        offset_y = 0.05 * np.cos(t * np.pi * 2) if motion_type == 'circle' else -0.1 * t

        rh = _generate_canonical_hand_pose(wrist_x=0.6 + offset_x, wrist_y=0.5 + offset_y, noise_std=noise_std, rng=rng)
        if two_handed:
            lh = _generate_canonical_hand_pose(wrist_x=0.4 - offset_x, wrist_y=0.5 + offset_y, noise_std=noise_std, rng=rng)
        else:
            lh = np.zeros(FEATURES_PER_HAND, dtype=np.float32)

        frames.append(np.concatenate([lh, rh]).tolist())

    return frames


def generate_synthetic_dataset(
    language: Union[str, SignLanguageEnum] = SignLanguageEnum.ASL,
    samples_per_class: int = 10,
    seed: int = 42,
    overwrite: bool = False,
) -> Dict[str, Any]:
    lang_enum = SignLanguageEnum(language) if isinstance(language, str) else language
    rng = np.random.RandomState(seed)

    if lang_enum == SignLanguageEnum.ASL:
        adapter = ASLDatasetAdapter()
        static_classes = ['Hello', 'Thank_You', 'Yes', 'No', 'Peace']
        dynamic_classes = ['Please', 'Sorry', 'Help', 'Friend', 'Book']
    elif lang_enum == SignLanguageEnum.ISL:
        adapter = ISLDatasetAdapter()
        static_classes = ['Namaste', 'Water', 'Good', 'Bad', 'Victory']
        dynamic_classes = ['Dance', 'School', 'Home', 'Family', 'Play']
    else:
        adapter = BSLDatasetAdapter()
        static_classes = ['Cheers', 'Tea', 'True', 'False', 'OK']
        dynamic_classes = ['Walk', 'Work', 'Learn', 'Meet', 'Goodbye']

    if overwrite:
        if adapter.static_file.exists(): adapter.static_file.unlink(missing_ok=True)
        if adapter.dynamic_file.exists(): adapter.dynamic_file.unlink(missing_ok=True)
        adapter._seen_hashes.clear()

    created_s = 0
    created_d = 0

    for label in static_classes:
        for _ in range(samples_per_class):
            feats = generate_static_sample_features(label, two_handed=(label in ['Namaste', 'Book', 'Family']), rng=rng)
            ok, _ = adapter.save_static_sample(feats, label=label)
            if ok: created_s += 1

    for label in dynamic_classes:
        for _ in range(samples_per_class):
            frames = generate_dynamic_sequence_features(label, two_handed=(label in ['Dance', 'School', 'Family', 'Book']), rng=rng)
            ok, _ = adapter.save_dynamic_sample(frames, label=label)
            if ok: created_d += 1

    return {
        'success': True,
        'language': lang_enum.value,
        'static_imported': created_s,
        'dynamic_imported': created_d,
        'total_samples': created_s + created_d,
        'total_labels': len(static_classes) + len(dynamic_classes),
        'static_classes': static_classes,
        'dynamic_classes': dynamic_classes,
    }
