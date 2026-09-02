"""
src/api/routes_custom.py
------------------------
API endpoints for Personalized Custom Signs ("Teach My Sign").
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Path as FPath, Query

from src.config import MIN_CUSTOM_SAMPLES_REQUIRED, RECOMMENDED_CUSTOM_SAMPLES
from src.custom_signs.manager import get_custom_sign_manager
from src.landmarks.extractor import extract_from_base64
from src.landmarks.normalizer import normalize_landmarks
from src.landmarks.sequence import compute_motion_energy, resample_sequence
from src.models.schemas import (
    CustomSignAddSamplesRequest,
    CustomSignCreateRequest,
    CustomSignRecord,
    CustomSignSampleInput,
    CustomSignUpdateRequest,
    SampleTypeEnum,
)

router = APIRouter(prefix="/api", tags=["Custom Signs"])


@router.post("/custom-sign/start")
async def start_custom_sign_session() -> Dict[str, Any]:
    return {
        "status": "ready",
        "instructions": "Enter a label, record sign 3 to 5 times, and click Save.",
        "min_samples_required": MIN_CUSTOM_SAMPLES_REQUIRED,
        "recommended_samples": RECOMMENDED_CUSTOM_SAMPLES,
    }


@router.post("/custom-sign/sample", response_model=CustomSignSampleInput)
async def process_custom_sign_sample(sample_payload: Dict[str, Any]) -> CustomSignSampleInput:
    b64_frames = sample_payload.get("b64_frames", [])
    raw_frames = sample_payload.get("frames", [])
    raw_features = sample_payload.get("features", [])

    if b64_frames:
        extracted = []
        for b64 in b64_frames:
            lms_126, _ = extract_from_base64(b64)
            extracted.append(lms_126.tolist())

        if len(extracted) == 1:
            return CustomSignSampleInput(
                sample_type=SampleTypeEnum.STATIC,
                features=extracted[0],
                motion_energy=0.0,
            )
        elif len(extracted) > 1:
            resampled = resample_sequence(extracted)
            motion = compute_motion_energy(resampled)
            return CustomSignSampleInput(
                sample_type=SampleTypeEnum.DYNAMIC,
                frames=[f.tolist() for f in resampled],
                motion_energy=motion,
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to extract landmarks.")

    elif raw_frames and len(raw_frames) > 1:
        resampled = resample_sequence(raw_frames)
        norm_frames = [normalize_landmarks(f).tolist() for f in resampled]
        motion = compute_motion_energy(resampled)
        return CustomSignSampleInput(
            sample_type=SampleTypeEnum.DYNAMIC,
            frames=norm_frames,
            motion_energy=motion,
        )

    elif raw_features or (raw_frames and len(raw_frames) == 1):
        target = raw_features if raw_features else raw_frames[0]
        norm_feats = normalize_landmarks(target).tolist()
        return CustomSignSampleInput(
            sample_type=SampleTypeEnum.STATIC,
            features=norm_feats,
            motion_energy=0.0,
        )

    else:
        raise HTTPException(status_code=400, detail="Invalid payload. Provide 'frames', 'features', or 'b64_frames'.")


@router.post("/custom-sign/save", response_model=CustomSignRecord, status_code=201)
async def save_custom_sign(request: CustomSignCreateRequest) -> CustomSignRecord:
    manager = get_custom_sign_manager()
    try:
        return manager.create_sign(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save custom sign: {str(e)}")


@router.get("/custom-signs", response_model=List[CustomSignRecord])
async def list_user_custom_signs(user_id: Optional[str] = Query(default=None)) -> List[CustomSignRecord]:
    manager = get_custom_sign_manager()
    return manager.list_signs(user_id=user_id)


@router.get("/custom-sign/{sign_id}", response_model=CustomSignRecord)
async def get_single_custom_sign(sign_id: str = FPath(...)) -> CustomSignRecord:
    manager = get_custom_sign_manager()
    sign = manager.get_sign(sign_id)
    if not sign:
        raise HTTPException(status_code=404, detail=f"Custom sign '{sign_id}' not found.")
    return sign


@router.put("/custom-sign/{sign_id}", response_model=CustomSignRecord)
async def update_single_custom_sign(
    request: CustomSignUpdateRequest,
    sign_id: str = FPath(...),
) -> CustomSignRecord:
    manager = get_custom_sign_manager()
    sign = manager.update_sign(sign_id, request)
    if not sign:
        raise HTTPException(status_code=404, detail=f"Custom sign '{sign_id}' not found.")
    return sign


@router.post("/custom-sign/{sign_id}/samples", response_model=CustomSignRecord)
async def add_samples_to_sign(
    request: CustomSignAddSamplesRequest,
    sign_id: str = FPath(...),
) -> CustomSignRecord:
    manager = get_custom_sign_manager()
    try:
        sign = manager.add_samples(sign_id, request.samples)
        if not sign:
            raise HTTPException(status_code=404, detail=f"Custom sign '{sign_id}' not found.")
        return sign
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/custom-sign/{sign_id}")
async def delete_single_custom_sign(sign_id: str = FPath(...)) -> Dict[str, Any]:
    manager = get_custom_sign_manager()
    deleted = manager.delete_sign(sign_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Custom sign '{sign_id}' not found.")
    return {"success": True, "message": f"Custom sign '{sign_id}' deleted successfully."}
