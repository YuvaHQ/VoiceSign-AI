"""
src/api/routes_recognition.py
-----------------------------
API routes and WebSocket endpoint for real-time sign recognition,
5-second persistent Help safety alarms, and Meeting Mode live transcript streaming.
"""

import json
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from src.landmarks.extractor import extract_from_base64
from src.landmarks.sequence import GestureSequenceBuffer
from src.models.schemas import (
    EventTypeEnum,
    RecognitionEvent,
    RecognitionRequest,
    RecognitionResult,
    SignLanguageEnum,
)
from src.recognition.debouncer import get_global_debouncer
from src.recognition.help_detector import get_global_help_detector
from src.recognition.hybrid_engine import get_hybrid_engine

router = APIRouter(prefix="/api/recognition", tags=["Recognition"])


@router.post("/frame", response_model=RecognitionResult)
async def recognize_single_frame(request: RecognitionRequest) -> RecognitionResult:
    if not request.features:
        raise HTTPException(status_code=400, detail="Missing 'features' in request payload.")

    engine = get_hybrid_engine()
    return engine.recognize_sign(request.features, language=request.language)


@router.post("/sequence", response_model=RecognitionResult)
async def recognize_frame_sequence(request: RecognitionRequest) -> RecognitionResult:
    if not request.sequence:
        raise HTTPException(status_code=400, detail="Missing 'sequence' in request payload.")

    engine = get_hybrid_engine()
    return engine.recognize_sign(request.sequence, language=request.language)


@router.get("/transcript")
async def get_current_transcript() -> Dict[str, Any]:
    debouncer = get_global_debouncer()
    return {
        "transcript": debouncer.get_transcript(),
        "recent_words": debouncer.get_recent_words(),
        "is_paused": debouncer.is_paused,
    }


@router.post("/transcript/clear")
async def clear_transcript() -> Dict[str, str]:
    debouncer = get_global_debouncer()
    debouncer.clear()
    return {"message": "Transcript cleared."}


@router.post("/transcript/pause")
async def pause_transcript() -> Dict[str, str]:
    debouncer = get_global_debouncer()
    debouncer.pause()
    return {"message": "Transcript paused."}


@router.post("/transcript/resume")
async def resume_transcript() -> Dict[str, str]:
    debouncer = get_global_debouncer()
    debouncer.resume()
    return {"message": "Transcript resumed."}


@router.post("/help/reset")
async def reset_help_alarm() -> Dict[str, str]:
    detector = get_global_help_detector()
    detector.reset()
    return {"message": "Help safety detector reset."}


@router.websocket("/ws")
async def websocket_recognition_stream(websocket: WebSocket):
    await websocket.accept()
    buffer = GestureSequenceBuffer()
    engine = get_hybrid_engine()
    current_language = SignLanguageEnum.ASL
    user_id = "default_user"

    try:
        while True:
            raw_msg = await websocket.receive_text()
            try:
                msg = json.loads(raw_msg)
            except Exception:
                continue

            msg_type = msg.get("type", "landmarks")
            if "language" in msg:
                try:
                    current_language = SignLanguageEnum(msg["language"])
                except ValueError:
                    pass
            if "user_id" in msg:
                user_id = msg["user_id"]

            if msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            landmarks_126 = None
            if msg_type == "landmarks":
                landmarks_126 = msg.get("data")
            elif msg_type == "image":
                b64 = msg.get("data", "")
                if b64:
                    lms_arr, _ = extract_from_base64(b64)
                    landmarks_126 = lms_arr.tolist()

            if landmarks_126 and len(landmarks_126) == 126:
                buffer.push(landmarks_126)

                res, events = engine.process_sequence_buffer(
                    buffer=buffer,
                    language=current_language,
                    user_id=user_id,
                )

                for ev in events:
                    await websocket.send_text(json.dumps(ev.model_dump()))

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
