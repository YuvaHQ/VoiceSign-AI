"""
tests/test_landmarks.py
-----------------------
Automated tests for landmark extraction, canonical representation,
spatial/temporal normalization, and sequence resampling.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from src.config import FEATURES_PER_HAND, SEQUENCE_LENGTH, TOTAL_FRAME_FEATURES
from src.landmarks.extractor import extract_canonical_landmarks
from src.landmarks.normalizer import center_and_scale_hand, normalize_landmarks, validate_landmarks
from src.landmarks.sequence import (
    GestureSequenceBuffer,
    compute_motion_energy,
    create_dynamic_sequence,
    create_static_sample,
    resample_sequence,
)
from src.models.schemas import SignLanguageEnum


def test_extract_canonical_landmarks_mock():
    mock_results = MagicMock()

    lm_left = [MagicMock(x=0.1, y=0.2, z=0.3) for _ in range(21)]
    hand_lms_left = MagicMock(landmark=lm_left)
    handedness_left = MagicMock(classification=[MagicMock(label="Left")])

    lm_right = [MagicMock(x=0.7, y=0.8, z=0.9) for _ in range(21)]
    hand_lms_right = MagicMock(landmark=lm_right)
    handedness_right = MagicMock(classification=[MagicMock(label="Right")])

    mock_results.multi_hand_landmarks = [hand_lms_left, hand_lms_right]
    mock_results.multi_handedness = [handedness_left, handedness_right]

    arr_126, detected = extract_canonical_landmarks(mock_results)
    assert arr_126.shape == (126,)
    assert set(detected) == {"Left", "Right"}
    assert arr_126[0] == 0.1
    assert arr_126[63] == 0.7


def test_extract_canonical_landmarks_single_hand_padding():
    mock_results = MagicMock()
    lm_right = [MagicMock(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    hand_lms_right = MagicMock(landmark=lm_right)
    handedness_right = MagicMock(classification=[MagicMock(label="Right")])

    mock_results.multi_hand_landmarks = [hand_lms_right]
    mock_results.multi_handedness = [handedness_right]

    arr_126, detected = extract_canonical_landmarks(mock_results)
    assert arr_126.shape == (126,)
    assert detected == ["Right"]
    assert np.all(arr_126[:FEATURES_PER_HAND] == 0.0)


def test_validate_and_normalize_landmarks():
    valid_vec = [0.5] * 126
    ok, _ = validate_landmarks(valid_vec)
    assert ok is True

    bad_size = [0.5] * 100
    ok, _ = validate_landmarks(bad_size)
    assert ok is False

    nan_vec = np.full(126, np.nan)
    norm = normalize_landmarks(nan_vec)
    assert not np.isnan(norm).any()
    assert np.all(norm == 0.0)


def test_sequence_resampling():
    short_seq = np.ones((10, 126), dtype=np.float32)
    resampled = resample_sequence(short_seq, target_length=30)
    assert resampled.shape == (30, 126)
    assert np.allclose(resampled, 1.0)

    long_seq = np.ones((45, 126), dtype=np.float32)
    resampled_long = resample_sequence(long_seq, target_length=30)
    assert resampled_long.shape == (30, 126)


def test_gesture_sequence_buffer():
    buf = GestureSequenceBuffer(sequence_length=30)
    assert len(buf) == 0
    assert not buf.is_ready()

    for i in range(30):
        buf.push([float(i)] * 126)

    assert len(buf) == 30
    assert buf.is_ready()
    assert buf.get_sequence_matrix().shape == (30, 126)
