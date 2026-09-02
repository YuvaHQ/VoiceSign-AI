"""
src/landmarks/normalizer.py
---------------------------
Spatial and temporal normalization of 126-dimensional hand landmark vectors.
"""

from typing import List, Tuple, Union
import numpy as np

from src.config import FEATURES_PER_HAND, TOTAL_FRAME_FEATURES


def validate_landmarks(landmarks_126: Union[List[float], np.ndarray]) -> Tuple[bool, str]:
    arr = np.asarray(landmarks_126, dtype=np.float32).flatten()
    if arr.shape[0] != TOTAL_FRAME_FEATURES:
        return False, f"Invalid landmarks: Expected dimensions {TOTAL_FRAME_FEATURES}, got {arr.shape[0]}"
    if np.isnan(arr).any():
        return False, "Landmarks contain NaN"
    if np.isinf(arr).any():
        return False, "Landmarks contain Inf"
    if (np.abs(arr) > 10.0).any():
        return False, "Landmarks out of range (> 10.0)"
    return True, "Valid"


def center_and_scale_hand(coords_63: Union[List[float], np.ndarray]) -> np.ndarray:
    arr = np.asarray(coords_63, dtype=np.float32).flatten()
    if not np.any(arr):
        return arr
    pts = arr.reshape(21, 3)
    wrist = pts[0].copy()
    pts = pts - wrist
    dists = np.linalg.norm(pts, axis=1)
    max_dist = float(np.max(dists))
    if max_dist > 1e-4:
        pts = pts / max_dist
    return pts.flatten()


def normalize_landmarks(
    landmarks_126: Union[List[float], np.ndarray],
    center_wrist: bool = False,
    scale_invariance: bool = False,
) -> np.ndarray:
    arr = np.asarray(landmarks_126, dtype=np.float32).flatten()
    if arr.shape[0] < TOTAL_FRAME_FEATURES:
        padded = np.zeros(TOTAL_FRAME_FEATURES, dtype=np.float32)
        padded[:min(arr.shape[0], TOTAL_FRAME_FEATURES)] = arr[:min(arr.shape[0], TOTAL_FRAME_FEATURES)]
        arr = padded
    elif arr.shape[0] > TOTAL_FRAME_FEATURES:
        arr = arr[:TOTAL_FRAME_FEATURES]

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if not center_wrist and not scale_invariance:
        return arr

    lh = arr[:FEATURES_PER_HAND]
    rh = arr[FEATURES_PER_HAND:]

    if np.any(lh):
        lh = center_and_scale_hand(lh)
    if np.any(rh):
        rh = center_and_scale_hand(rh)

    return np.concatenate([lh, rh]).astype(np.float32)
