"""
src/landmarks/extractor.py
--------------------------
Extracts 126-dimensional canonical hand landmarks from images and frames.
"""

import base64
import os
from typing import Any, List, Optional, Tuple
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

from src.config import BASE_DIR, FEATURES_PER_HAND, TOTAL_FRAME_FEATURES

DEFAULT_TASK_MODEL_PATH = str(BASE_DIR / 'hand_landmarker.task')


def extract_canonical_landmarks(results) -> Tuple[np.ndarray, List[str]]:
    lh = [0.0] * FEATURES_PER_HAND
    rh = [0.0] * FEATURES_PER_HAND
    detected = []

    if hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks is not None:
        hand_landmarks_list = results.multi_hand_landmarks
        handedness_list = getattr(results, 'multi_handedness', None)
    else:
        hand_landmarks_list = getattr(results, 'hand_landmarks', None)
        handedness_list = getattr(results, 'handedness', None)

    if hand_landmarks_list and handedness_list:
        for hand_lms, handedness in zip(hand_landmarks_list, handedness_list):
            if hasattr(handedness, 'classification') and handedness.classification:
                label = getattr(handedness.classification[0], 'label', 'Unknown')
            elif isinstance(handedness, list) and len(handedness) > 0:
                cat = handedness[0]
                label = (
                    getattr(cat, 'category_name', None)
                    or getattr(cat, 'display_name', None)
                    or getattr(cat, 'label', 'Unknown')
                )
            else:
                label = 'Unknown'

            lms = getattr(hand_lms, 'landmark', hand_lms)
            coords = []
            for lm in lms:
                coords.extend([round(float(lm.x), 6), round(float(lm.y), 6), round(float(lm.z), 6)])

            if label == 'Left':
                lh = coords
                detected.append('Left')
            elif label == 'Right':
                rh = coords
                detected.append('Right')
            else:
                if not any(lh):
                    lh = coords
                    detected.append('Hand1')
                else:
                    rh = coords
                    detected.append('Hand2')

    arr = np.asarray(lh + rh, dtype=np.float32)
    return arr, detected


class MediaPipeLandmarkExtractor:
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: Optional[str] = None,
    ):
        self.model_path = model_path or DEFAULT_TASK_MODEL_PATH
        base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

    def extract_from_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        if frame_bgr is None or frame_bgr.size == 0:
            return np.zeros(TOTAL_FRAME_FEATURES, dtype=np.float32), []
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.detector.detect(mp_image)
        return extract_canonical_landmarks(results)

    def close(self):
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.close()


_GLOBAL_EXTRACTOR: Optional[MediaPipeLandmarkExtractor] = None


def get_landmark_extractor() -> MediaPipeLandmarkExtractor:
    global _GLOBAL_EXTRACTOR
    if _GLOBAL_EXTRACTOR is None:
        _GLOBAL_EXTRACTOR = MediaPipeLandmarkExtractor()
    return _GLOBAL_EXTRACTOR


def extract_from_base64(base64_image_str: str) -> Tuple[np.ndarray, List[str]]:
    try:
        if "," in base64_image_str:
            base64_image_str = base64_image_str.split(",")[1]
        img_bytes = base64.b64decode(base64_image_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return np.zeros(TOTAL_FRAME_FEATURES, dtype=np.float32), []
        extractor = get_landmark_extractor()
        return extractor.extract_from_frame(frame)
    except Exception:
        return np.zeros(TOTAL_FRAME_FEATURES, dtype=np.float32), []