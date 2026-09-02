"""
src/services/gemini_service.py
------------------------------
AI service powered by Google GenAI SDK (gemini-2.5-flash).
Provides:
  1. Zero-shot automatic ASL and ISL sign recognition from landmarks and/or video frames.
  2. Robust 3D orientation-invariant geometric feature extraction and offline kinematic trajectory classification.
  3. Natural-language sentence restructuring & grammar polishing from sign glosses.
  4. Generating structured explanations and descriptions of custom signs.
"""

import base64
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.config import GEMINI_API_KEY, GEMINI_MODEL_NAME

try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


def _analyze_hand_geometry(lms_63: Union[List[float], np.ndarray]) -> Optional[Dict[str, Any]]:
    """
    Extracts orientation-invariant 3D finger extension, spacing, angles, and handshape characteristics.
    """
    pts = np.asarray(lms_63, dtype=np.float32).reshape(21, 3)
    if not np.any(pts != 0):
        return None

    wrist = pts[0]

    # Invariant Finger Extension & Straightness
    # Tips: 4 (Thumb), 8 (Index), 12 (Middle), 16 (Ring), 20 (Pinky)
    # PIPs: 2 (Thumb MCP), 6 (Index PIP), 10 (Middle PIP), 14 (Ring PIP), 18 (Pinky PIP)
    # MCPs: 1 (Thumb CMC), 5 (Index MCP), 9 (Middle MCP), 13 (Ring MCP), 17 (Pinky MCP)
    def _is_ext(tip_i, pip_i, mcp_i, min_len=0.085):
        d_tip_wrist = float(np.linalg.norm(pts[tip_i] - wrist))
        d_mcp_wrist = float(np.linalg.norm(pts[mcp_i] - wrist))
        d_tip_mcp = float(np.linalg.norm(pts[tip_i] - pts[mcp_i]))
        d_pip_mcp = float(np.linalg.norm(pts[pip_i] - pts[mcp_i]))
        d_tip_pip = float(np.linalg.norm(pts[tip_i] - pts[pip_i]))
        straightness = d_tip_mcp / (d_pip_mcp + d_tip_pip + 1e-6)
        return bool(d_tip_wrist > d_mcp_wrist * 1.25 and straightness > 0.88 and d_tip_mcp >= min_len)

    d_thumb_tip = float(np.linalg.norm(pts[4] - wrist))
    d_thumb_mcp = float(np.linalg.norm(pts[2] - wrist))
    d_thumb_idx_mcp = float(np.linalg.norm(pts[4] - pts[5]))
    thumb_ext = bool(d_thumb_tip > d_thumb_mcp * 1.22 and d_thumb_idx_mcp > 0.08)

    idx_ext = _is_ext(8, 6, 5)
    mid_ext = _is_ext(12, 10, 9)
    ring_ext = _is_ext(16, 14, 13)
    pinky_ext = _is_ext(20, 18, 17)

    # Inter-finger distances & Pinches
    dist_thumb_idx = float(np.linalg.norm(pts[4] - pts[8]))
    dist_thumb_mid = float(np.linalg.norm(pts[4] - pts[12]))
    dist_idx_mid = float(np.linalg.norm(pts[8] - pts[12]))
    dist_mid_ring = float(np.linalg.norm(pts[12] - pts[16]))
    dist_ring_pinky = float(np.linalg.norm(pts[16] - pts[20]))

    d_idx_tip = float(np.linalg.norm(pts[8] - wrist))
    d_idx_mcp = float(np.linalg.norm(pts[5] - wrist))
    is_ok_pinch = bool(dist_thumb_idx < 0.065 and not idx_ext and mid_ext and ring_ext and pinky_ext)
    is_o_shape = bool(d_idx_tip > d_idx_mcp * 1.15 and dist_thumb_idx < 0.08 and dist_thumb_mid < 0.08 and not idx_ext and not mid_ext and not ring_ext and not pinky_ext)
    is_u_together = bool(idx_ext and mid_ext and dist_idx_mid < 0.048 and not ring_ext and not pinky_ext)
    is_v_apart = bool(idx_ext and mid_ext and dist_idx_mid >= 0.048 and not ring_ext and not pinky_ext)
    is_w_shape = bool(idx_ext and mid_ext and ring_ext and not pinky_ext)
    # Finger Crossing (e.g. Middle crossed over Index for Letter R)
    mcp_vec = pts[9] - pts[5]
    tip_vec = pts[12] - pts[8]
    is_crossed_r = bool(idx_ext and mid_ext and dist_idx_mid < 0.05 and float(np.dot(mcp_vec, tip_vec)) < -1e-4)

    ext_count = sum([thumb_ext, idx_ext, mid_ext, ring_ext, pinky_ext])

    # Classify Handshape
    if is_ok_pinch:
        shape = "OK-Sign / Letter F / Number 9"
    elif is_o_shape:
        shape = "O-Shape / Letter O"
    elif is_crossed_r:
        shape = "Crossed Fingers / Letter R"
    elif is_u_together:
        shape = "U-Shape (Fingers Together) / Letter U / Letter H"
    elif is_v_apart:
        shape = "V-Shape / Peace / Number 2 / Letter V"
    elif is_w_shape:
        shape = "W-Shape / Water / Number 3 / Letter W"
    elif ext_count == 0:
        if pts[4, 1] < pts[6, 1]:  # Thumb upright
            shape = "Fist (Letter A)"
        elif pts[4, 0] > pts[8, 0] and pts[4, 0] < pts[16, 0]:
            shape = "Fist with Thumb Across (Letter S)"
        else:
            shape = "Fist / Closed Hand (A / S / E / M / N / T)"
    elif ext_count == 5:
        shape = "Open Palm / 5-Hand (B / 5 / Hello / Stop)"
    elif idx_ext and not (thumb_ext or mid_ext or ring_ext or pinky_ext):
        shape = "Index Point / Number 1 / Letter D / Pointer"
    elif thumb_ext and not (idx_ext or mid_ext or ring_ext or pinky_ext):
        if pts[4, 1] < pts[2, 1]:
            shape = "Thumbs-Up / Good / Like / Letter A"
        else:
            shape = "Thumbs-Down / Bad / Dislike"
    elif pinky_ext and not (thumb_ext or idx_ext or mid_ext or ring_ext):
        shape = "Pinky Point / Letter I / Number 6"
    elif thumb_ext and idx_ext and not (mid_ext or ring_ext or pinky_ext):
        shape = "L-Shape / Letter L"
    elif thumb_ext and pinky_ext and not (idx_ext or mid_ext or ring_ext):
        shape = "Y-Shape / Letter Y / Phone / Same"
    elif thumb_ext and idx_ext and pinky_ext and not (mid_ext or ring_ext):
        shape = "I Love You (ILY)"
    elif idx_ext and pinky_ext and not (thumb_ext or mid_ext or ring_ext):
        shape = "Horns / Rock-On"
    elif not thumb_ext and idx_ext and mid_ext and ring_ext and pinky_ext:
        shape = "Four Fingers / Letter B / Number 4"
    else:
        shape = f"{ext_count} fingers extended"

    return {
        "wrist_position": [round(float(wrist[0]), 3), round(float(wrist[1]), 3), round(float(wrist[2]), 3)],
        "finger_status": {
            "thumb": "Extended" if thumb_ext else "Folded",
            "index": "Extended" if idx_ext else "Folded",
            "middle": "Extended" if mid_ext else "Folded",
            "ring": "Extended" if ring_ext else "Folded",
            "pinky": "Extended" if pinky_ext else "Folded",
        },
        "hand_shape": shape,
        "is_ok_pinch": is_ok_pinch,
        "is_u_together": is_u_together,
        "is_v_apart": is_v_apart,
        "is_w_shape": is_w_shape,
        "thumb_ext": thumb_ext,
        "idx_ext": idx_ext,
        "mid_ext": mid_ext,
        "ring_ext": ring_ext,
        "pinky_ext": pinky_ext,
    }


def _analyze_dynamic_trajectory(frames: Union[List[List[float]], np.ndarray]) -> Optional[Dict[str, Any]]:
    """
    Extracts trajectory kinematics, velocity reversals, enclosed shoelace area,
    and two-hand interaction dynamics from a temporal sequence of landmark frames.
    """
    seq = np.asarray(frames, dtype=np.float32).reshape(-1, 126)
    n = len(seq)
    if n < 3:
        return None

    f4d = seq.reshape(n, 2, 21, 3)

    lh_active = bool(np.any(f4d[:, 0, :, :] != 0))
    rh_active = bool(np.any(f4d[:, 1, :, :] != 0))

    rh_motion = float(np.std(f4d[:, 1, 0, :], axis=0).mean()) if rh_active else 0.0
    lh_motion = float(np.std(f4d[:, 0, 0, :], axis=0).mean()) if lh_active else 0.0

    dom_idx = 1 if rh_motion >= lh_motion else 0
    dom_label = "Right" if dom_idx == 1 else "Left"

    wrist_traj = f4d[:, dom_idx, 0, :]
    deltas = np.diff(wrist_traj, axis=0)
    path_len = float(np.sum(np.linalg.norm(deltas, axis=1)))
    net_dx = float(wrist_traj[-1, 0] - wrist_traj[0, 0])
    net_dy = float(wrist_traj[-1, 1] - wrist_traj[0, 1])
    net_disp = float(np.linalg.norm(wrist_traj[-1] - wrist_traj[0]))

    vx = deltas[:, 0]
    vy = deltas[:, 1]
    dx_flips = int(np.sum(np.diff(np.sign(vx[np.abs(vx) > 0.001])) != 0))
    dy_flips = int(np.sum(np.diff(np.sign(vy[np.abs(vy) > 0.001])) != 0))

    x = wrist_traj[:, 0]
    y = wrist_traj[:, 1]
    area = 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
    circularity = float(abs(4 * np.pi * area) / (path_len**2 + 1e-6)) if path_len > 0.05 else 0.0

    two_handed = lh_active and rh_active
    dist_wrists_start = float(np.linalg.norm(f4d[0, 0, 0, :] - f4d[0, 1, 0, :])) if two_handed else None
    dist_wrists_end = float(np.linalg.norm(f4d[-1, 0, 0, :] - f4d[-1, 1, 0, :])) if two_handed else None

    lh_dx = float(f4d[-1, 0, 0, 0] - f4d[0, 0, 0, 0]) if lh_active else 0.0
    rh_dx = float(f4d[-1, 1, 0, 0] - f4d[0, 1, 0, 0]) if rh_active else 0.0

    return {
        "num_frames": n,
        "dominant_hand": dom_label,
        "path_length": round(path_len, 4),
        "net_dx": round(net_dx, 4),
        "net_dy": round(net_dy, 4),
        "net_disp": round(net_disp, 4),
        "dx_flips": dx_flips,
        "dy_flips": dy_flips,
        "circularity": round(circularity, 4),
        "two_handed": two_handed,
        "dist_wrists_start": round(dist_wrists_start, 3) if dist_wrists_start is not None else None,
        "dist_wrists_end": round(dist_wrists_end, 3) if dist_wrists_end is not None else None,
        "lh_dx": round(lh_dx, 4),
        "rh_dx": round(rh_dx, 4),
    }


class GeminiService:
    """Service wrapper for Google Gemini multimodal sign language recognition & assistants."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = GEMINI_MODEL_NAME):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "") or GEMINI_API_KEY
        self.model_name = model_name
        self._client = None
        if _GENAI_AVAILABLE and self.api_key:
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception:
                self._client = None

    def is_available(self) -> bool:
        """Check if Gemini API is configured and accessible."""
        return _GENAI_AVAILABLE and bool(self.api_key) and self._client is not None

    def polish_sentence(self, words: List[str]) -> Tuple[str, bool, float]:
        start = time.time()
        if not words:
            return "", False, 0.0

        raw_str = " ".join(words).strip()
        fallback_str = raw_str.capitalize() + ("." if not raw_str.endswith((".", "!", "?")) else "")

        if not self.is_available():
            latency = round(time.time() - start, 4)
            return fallback_str, False, latency

        prompt = f"""You are a Sign Language interpreter. Transform these sign language glosses/words into a clear, natural, grammatically correct English sentence.
Preserve the exact meaning. Do not add unnecessary new details.

Sign glosses: {raw_str}

Respond ONLY with the polished English sentence."""

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            polished = response.text.strip()
            latency = round(time.time() - start, 4)
            return polished, True, latency
        except Exception:
            latency = round(time.time() - start, 4)
            return fallback_str, False, latency

    def describe_sign(self, label: str, language: str = "ASL") -> Tuple[str, bool]:
        fallback = f"{language} sign for {label}."
        if not self.is_available():
            return fallback, False

        prompt = f"Provide a 1-sentence physical description of how to perform the {language} sign for '{label}'."
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response.text.strip(), True
        except Exception:
            return fallback, False

    def recognize_sign(
        self,
        data: Union[List[float], List[List[float]], np.ndarray],
        language: str = "AUTO",
        is_dynamic: Optional[bool] = None,
        image_bgr: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Automatically detects ASL or ISL sign from landmark metrics and/or video frames
        without requiring pre-trained local ML models or training datasets.
        """
        arr = np.asarray(data, dtype=np.float32)

        # Detect motion dynamics if not explicitly supplied
        if is_dynamic is None:
            if arr.ndim == 2 or (arr.ndim == 1 and arr.shape[0] > 126):
                seq = arr.reshape(-1, 126)
                var = np.std(seq, axis=0).mean()
                is_dynamic = bool(var >= 0.038)
            else:
                is_dynamic = False

        flat_126 = arr.flatten()[:126]
        lh_geom = _analyze_hand_geometry(flat_126[:63])
        rh_geom = _analyze_hand_geometry(flat_126[63:])
        active_geom = rh_geom or lh_geom or {}

        dist_wrists = None
        dist_idx_tips = None
        if lh_geom and rh_geom:
            lh_pts = flat_126[:63].reshape(21, 3)
            rh_pts = flat_126[63:].reshape(21, 3)
            dist_wrists = round(float(np.linalg.norm(lh_pts[0] - rh_pts[0])), 3)
            dist_idx_tips = round(float(np.linalg.norm(lh_pts[8] - rh_pts[8])), 3)

        traj = None
        if is_dynamic and arr.size >= 126:
            frames = arr.reshape(-1, 126)
            traj = _analyze_dynamic_trajectory(frames)
            num_frames = len(frames)
            start_lh = _analyze_hand_geometry(frames[0, :63])
            start_rh = _analyze_hand_geometry(frames[0, 63:])
            mid_lh = _analyze_hand_geometry(frames[num_frames // 2, :63])
            mid_rh = _analyze_hand_geometry(frames[num_frames // 2, 63:])
            end_lh = _analyze_hand_geometry(frames[-1, :63])
            end_rh = _analyze_hand_geometry(frames[-1, 63:])

            motion_summary = f"Trajectory: dominant={traj['dominant_hand']}, path_len={traj['path_length']}, net_dx={traj['net_dx']:+.3f}, net_dy={traj['net_dy']:+.3f}, x_oscillations={traj['dx_flips']}, y_oscillations={traj['dy_flips']}, circularity={traj['circularity']}"

            geom_text = f"""DYNAMIC GESTURE SEQUENCE:
{motion_summary}
Start Posture: Left={start_lh.get('hand_shape') if start_lh else 'None'}, Right={start_rh.get('hand_shape') if start_rh else 'None'}
Mid Posture:   Left={mid_lh.get('hand_shape') if mid_lh else 'None'}, Right={mid_rh.get('hand_shape') if mid_rh else 'None'}
End Posture:   Left={end_lh.get('hand_shape') if end_lh else 'None'}, Right={end_rh.get('hand_shape') if end_rh else 'None'}"""
        else:
            geom_text = f"""STATIC GESTURE POSE:
Left Hand: {lh_geom.get('hand_shape') if lh_geom else 'None'}
Right Hand: {rh_geom.get('hand_shape') if rh_geom else 'None'}
Wrist Distance: {dist_wrists if dist_wrists is not None else 'N/A'}
Index Tips Distance: {dist_idx_tips if dist_idx_tips is not None else 'N/A'}"""

        # Offline High-Accuracy Heuristic Recognizer
        def _heuristic_classify() -> Tuple[str, str, float, str]:
            if not active_geom and not lh_geom and not rh_geom:
                return "Waiting for hands...", "AUTO", 0.0, "No hands in frame."

            # Dynamic Gesture Classification
            if is_dynamic and traj:
                dom_shape = active_geom.get("hand_shape", "")
                
                # Two-handed dynamic signs
                if traj["two_handed"]:
                    # Help: Fist resting on or lifting upward from flat palm
                    is_one_fist = any("Fist" in str(g.get("hand_shape")) or "Closed" in str(g.get("hand_shape")) for g in [lh_geom, rh_geom] if g)
                    is_one_open = any("Open" in str(g.get("hand_shape")) or "Four" in str(g.get("hand_shape")) or "5" in str(g.get("hand_shape")) or "B" in str(g.get("hand_shape")) for g in [lh_geom, rh_geom] if g)
                    if (is_one_fist or is_one_open) and traj["net_dy"] < -0.04:
                        return "Help", "ASL", 0.95, "Fist resting on flat palm lifting upward (Help sign)."
                    # Welcome: Both open palms sweeping inward
                    if traj["lh_dx"] > 0.03 and traj["rh_dx"] < -0.03:
                        return "Welcome", "ISL", 0.94, "Both open palms sweeping inward toward torso."
                    # Dance (ISL): Two fingers moving on flat palm
                    if ("V-Shape" in str(lh_geom.get("hand_shape")) or "V-Shape" in str(rh_geom.get("hand_shape"))) and ("Open Palm" in str(lh_geom.get("hand_shape")) or "Open Palm" in str(rh_geom.get("hand_shape"))):
                        return "Dance", "ISL", 0.94, "Two fingers moving across opposite flat palm."
                    # Book: Two hands opening
                    if "Open Palm" in str(lh_geom.get("hand_shape")) and "Open Palm" in str(rh_geom.get("hand_shape")):
                        if traj["path_length"] > 0.08:
                            return "Book", "ASL", 0.92, "Two open palms opening apart."

                # Single dominant hand dynamic signs
                if "Open Palm" in dom_shape or "5-Hand" in dom_shape or "Four Fingers" in dom_shape:
                    # Hello / Wave: lateral oscillation
                    if traj["dx_flips"] >= 2 or abs(traj["net_dx"]) > 0.12:
                        return "Hello", "ASL", 0.95, "Open palm waving side-to-side in greeting."
                    # Thank You: forward stroke downward away from chin
                    if traj["net_dy"] > 0.05:
                        return "Thank You", "ASL", 0.95, "Open hand moving forward from chin toward camera."
                    # Please: circular chest motion
                    if traj["circularity"] > 0.40 or (traj["dx_flips"] >= 1 and traj["dy_flips"] >= 1):
                        return "Please", "ASL", 0.94, "Flat palm moving in circular motion on chest."

                if "Fist" in dom_shape:
                    # Sorry / Please: circular rubbing motion
                    if traj["circularity"] > 0.40 or (traj["dx_flips"] >= 1 and traj["dy_flips"] >= 1):
                        return "Sorry / Please", "ASL", 0.94, "Fist moving in circular motion on chest."
                    # Yes: vertical nodding
                    if traj["dy_flips"] >= 2 or (traj["net_dy"] > 0.06 and traj["path_length"] > 0.10):
                        return "Yes", "ASL", 0.94, "Fist bobbing/nodding up and down in agreement."

                if "Thumbs-Up" in dom_shape:
                    return "Good", "ASL", 0.94, "Thumbs-up gesture with forward motion."

                if "Thumbs-Down" in dom_shape:
                    return "Bad", "ASL", 0.94, "Thumbs-down gesture moving downward."

                if "W-Shape" in dom_shape:
                    return "Water", "ASL", 0.95, "W-hand tapping near chin."

                if "I Love You" in dom_shape or "ILY" in dom_shape:
                    return "I Love You", "ASL", 0.95, "ILY gesture moving in expression."

            # Static Pose Classification
            # Both hands present
            if lh_geom and rh_geom and dist_wrists is not None:
                if dist_wrists < 0.22 and dist_idx_tips is not None and dist_idx_tips < 0.15:
                    return "Namaste", "ISL", 0.95, "Both flat palms pressed vertically together in greeting."
                if dist_wrists < 0.35 and "Open Palm" in str(lh_geom.get("hand_shape")) and "Open Palm" in str(rh_geom.get("hand_shape")):
                    return "Book", "ASL", 0.92, "Two open palms held together like an open book."

            # Single active hand static
            h = active_geom
            shape = h.get("hand_shape", "")
            if "I Love You" in shape or "ILY" in shape:
                return "I Love You", "ASL", 0.95, "Thumb, index, and pinky extended (ILY sign)."
            if "OK-Sign" in shape:
                return "OK / F", "ASL", 0.94, "Thumb and index touching in circle with 3 fingers up."
            if "V-Shape" in shape:
                return "Peace / V", "ASL", 0.95, "Index and middle fingers extended apart in V-shape."
            if "W-Shape" in shape:
                return "Water / W", "ASL", 0.94, "Three middle fingers extended upright (W sign)."
            if "Thumbs-Up" in shape:
                return "Good / Like", "ASL", 0.95, "Thumb pointing straight up in approval."
            if "Thumbs-Down" in shape:
                return "Bad / Dislike", "ASL", 0.95, "Thumb pointing down in disapproval."
            if "L-Shape" in shape:
                return "L", "ASL", 0.95, "Thumb and index finger forming 90-degree L."
            if "Y-Shape" in shape:
                return "Y / Phone", "ASL", 0.94, "Thumb and pinky extended outward."
            if "Crossed Fingers" in shape:
                return "R", "ASL", 0.93, "Middle finger crossed over index finger."
            if "U-Shape" in shape:
                return "U", "ASL", 0.93, "Index and middle fingers extended tightly together."
            if "Index Point" in shape:
                return "D / 1", "ASL", 0.92, "Index finger pointing upright."
            if "Pinky Point" in shape:
                return "I", "ASL", 0.92, "Pinky finger pointing upright."
            if "Open Palm" in shape:
                return "Stop / 5", "ASL", 0.90, "Flat open palm facing forward."
            if "Fist" in shape:
                return "A / S", "ASL", 0.88, "Closed hand / solid fist."

            return "Neutral / No Sign", "AUTO", 0.0, "Hand detected but no deliberate sign formed."

        h_sign, h_lang, h_conf, h_exp = _heuristic_classify()

        if not self.is_available():
            target_l = language if language != "AUTO" else (h_lang if h_conf > 0 else "ASL")
            return {
                "sign": h_sign,
                "language": target_l,
                "confidence": h_conf,
                "mode": "DYNAMIC" if is_dynamic else "STATIC",
                "explanation": f"{h_exp} (Offline Kinematic Engine)",
                "is_gemini_generated": False,
            }

        prompt = f"""You are an expert real-time AI Sign Language Recognition engine for American Sign Language (ASL) and Indian Sign Language (ISL).
Analyze the spatial hand landmarks and determine the exact sign being performed.

Target Language: {language} (If AUTO, determine whether the gesture is from ASL or ISL)

LANDMARK 3D GEOMETRIC & KINEMATIC ANALYSIS:
{geom_text}

CANONICAL DYNAMIC & STATIC SIGN DEFINITIONS:
- ASL Dynamic Words:
  * Hello: Open 5-palm waving side-to-side (lateral oscillation).
  * Thank You: Flat hand moves forward and downward away from chin/mouth.
  * Please: Flat palm moving in circular motion on chest.
  * Sorry: Fist moving in circular rubbing motion on chest.
  * Help: Closed fist resting on flat palm and lifting upward together.
  * Yes: Closed fist bobbing/nodding up and down.
  * No: Index and middle fingers snapping down onto thumb.
  * Water: 'W' hand (3 fingers) tapped near chin/mouth.
  * I Love You: ILY hand moving forward in expression.
- ISL Dynamic Words:
  * Namaste: Two flat palms coming together and holding vertically in prayer.
  * Dance: Two fingers (V) moving/hopping across opposite flat palm.
  * Welcome: Both open palms sweeping inward toward torso.
  * Eat / Food: Bundled fingertips moving repeatedly to mouth.
  * Tea: O-hand dipping/moving near mouth.
- ASL Alphabet & Numbers (Static): A, B, C, D, E, F, G, H, I, K, L, O, R, S, U, V, W, Y, 1, 2, 3, 4, 5, Thumbs Up (Good), Thumbs Down (Bad).

CRITICAL ACCURACY RULES:
1. If the signer's hand is relaxed, neutral, transitioning, adjusting camera, or NOT clearly matching a deliberate sign from the glossary, you MUST return:
   "sign": "Neutral / No Sign", "confidence": 0.0, "explanation": "Hand in neutral or transitioning posture."
2. Do NOT guess or hallucinate a sign if the movement and finger configuration do not match the definition.
3. In AUTO mode, only specify ISL if the gesture is unique to ISL (e.g. Namaste, Dance, Welcome), otherwise default to ASL.

Return ONLY a valid JSON object in this exact format:
{{
  "sign": "<Exact sign name or 'Neutral / No Sign'>",
  "language": "<ASL, ISL, or AUTO>",
  "confidence": <float between 0.0 and 1.0>,
  "mode": "<STATIC or DYNAMIC>",
  "explanation": "<1 concise sentence explaining the matching hand shape, finger positions, and motion trajectory>"
}}
"""

        try:
            contents = [prompt]
            if image_bgr is not None and image_bgr.size > 0:
                ret, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    contents.append(types.Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg"))

            response = self._client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.10,
                ),
            )
            raw = response.text.strip()
            parsed = json.loads(raw)
            return {
                "sign": parsed.get("sign", h_sign),
                "language": parsed.get("language", language if language != "AUTO" else h_lang),
                "confidence": float(parsed.get("confidence", 0.92)),
                "mode": parsed.get("mode", "DYNAMIC" if is_dynamic else "STATIC"),
                "explanation": parsed.get("explanation", h_exp),
                "is_gemini_generated": True,
            }
        except Exception:
            return {
                "sign": h_sign,
                "language": h_lang,
                "confidence": h_conf,
                "mode": "DYNAMIC" if is_dynamic else "STATIC",
                "explanation": f"{h_exp} (Fallback Kinematic Engine)",
                "is_gemini_generated": False,
            }


_gemini_service_instance: Optional[GeminiService] = None


def get_gemini_service(api_key: Optional[str] = None) -> GeminiService:
    global _gemini_service_instance
    if _gemini_service_instance is None or api_key is not None:
        _gemini_service_instance = GeminiService(api_key=api_key)
    return _gemini_service_instance

        
