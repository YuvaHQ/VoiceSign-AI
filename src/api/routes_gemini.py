"""
src/api/routes_gemini.py
------------------------
API endpoints for Gemini sentence grammar polishing and sign descriptions.
"""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException

from src.models.schemas import (
    DescribeSignRequest,
    DescribeSignResponse,
    SentencePolishRequest,
    SentencePolishResponse,
)
from src.services.gemini_service import get_gemini_service

router = APIRouter(prefix="/api/gemini", tags=["Gemini AI Support"])


@router.get("/status")
async def get_gemini_status() -> Dict[str, Any]:
    service = get_gemini_service()
    return {
        "available": service.is_available(),
        "model_name": service.model_name,
        "features": [
            "Sign gloss to natural sentence translation",
            "Assistive sign formation descriptions",
            "Conversational meeting transcript polishing",
        ],
    }


@router.post("/polish-sentence", response_model=SentencePolishResponse)
async def polish_sentence(request: SentencePolishRequest) -> SentencePolishResponse:
    service = get_gemini_service()
    polished, is_gemini, latency = service.polish_sentence(request.words, context=request.context or "general")

    return SentencePolishResponse(
        original_glosses=request.words,
        polished_sentence=polished,
        is_gemini_generated=is_gemini,
        latency_ms=latency,
    )


@router.post("/describe-sign", response_model=DescribeSignResponse)
async def describe_sign(request: DescribeSignRequest) -> DescribeSignResponse:
    service = get_gemini_service()
    desc, is_gemini = service.describe_sign(request.label, language=request.language.value)

    return DescribeSignResponse(
        label=request.label,
        description=desc,
        is_gemini_generated=is_gemini,
    )
