"""
src/models/schemas.py
---------------------
Pydantic data schemas, models, and enums for Multilingual & Personalized Sign Language System.
"""

from datetime import datetime
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class SignLanguageEnum(str, Enum):
    ASL = 'ASL'
    ISL = 'ISL'
    BSL = 'BSL'
    CUSTOM = 'CUSTOM'


class SampleTypeEnum(str, Enum):
    STATIC = 'static'
    DYNAMIC = 'dynamic'


class EventTypeEnum(str, Enum):
    SIGN_RECOGNIZED = 'SIGN_RECOGNIZED'
    TRANSCRIPT_UPDATED = 'TRANSCRIPT_UPDATED'
    HELP_DETECTED = 'HELP_DETECTED'
    CUSTOM_SIGN_RECOGNIZED = 'CUSTOM_SIGN_RECOGNIZED'
    ERROR = 'ERROR'


class SampleMetadata(BaseModel):
    recorded_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    num_frames: Optional[int] = None
    motion_energy: Optional[float] = None
    detected_hands: Optional[List[str]] = None
    custom_properties: Optional[Dict[str, Any]] = None


class StaticSignSample(BaseModel):
    language: SignLanguageEnum
    label: str
    sample_type: SampleTypeEnum = SampleTypeEnum.STATIC
    source: str = 'user'
    features: List[float]  # 126 floats
    metadata: SampleMetadata = Field(default_factory=SampleMetadata)


class DynamicSignSample(BaseModel):
    language: SignLanguageEnum
    label: str
    sample_type: SampleTypeEnum = SampleTypeEnum.DYNAMIC
    source: str = 'user'
    frames: List[List[float]]  # List of 126-float lists
    metadata: SampleMetadata = Field(default_factory=SampleMetadata)


class CustomSignSampleInput(BaseModel):
    sample_type: SampleTypeEnum = SampleTypeEnum.STATIC
    features: Optional[List[float]] = None
    frames: Optional[List[List[float]]] = None
    motion_energy: Optional[float] = 0.0
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class CustomSignSample(BaseModel):
    sample_id: str
    sample_type: SampleTypeEnum
    features: Optional[List[float]] = None
    frames: Optional[List[List[float]]] = None
    motion_energy: Optional[float] = 0.0
    created_at: str


class CustomSign(BaseModel):
    id: str = Field(..., alias='id')
    user_id: str = 'default_user'
    label: str
    language: str = 'CUSTOM'
    description: Optional[str] = ''
    sample_type_summary: Optional[str] = 'mixed'
    samples: List[CustomSignSample] = []
    created_at: Union[float, str]
    updated_at: Union[float, str]
    sample_count: int = 0
    is_trained: bool = False

    @property
    def sign_id(self) -> str:
        return self.id


# Alias for CustomSign
CustomSignRecord = CustomSign


class CreateCustomSignRequest(BaseModel):
    user_id: str = 'default_user'
    label: str
    description: Optional[str] = ''
    sample_type: SampleTypeEnum = SampleTypeEnum.STATIC
    samples: List[CustomSignSampleInput] = []


# Alias
CustomSignCreateRequest = CreateCustomSignRequest


class UpdateCustomSignRequest(BaseModel):
    label: Optional[str] = None
    description: Optional[str] = None


# Alias
CustomSignUpdateRequest = UpdateCustomSignRequest


class AddSamplesRequest(BaseModel):
    samples: List[CustomSignSampleInput]


class CustomSignResponse(BaseModel):
    success: bool
    message: str
    sign: Optional[CustomSign] = None


class ListCustomSignsResponse(BaseModel):
    signs: List[CustomSign]
    total: int


class DatasetStatus(BaseModel):
    language: SignLanguageEnum
    sample_count: int = 0
    static_samples_count: int = 0
    dynamic_samples_count: int = 0
    distinct_labels_count: int = 0
    labels: List[str] = []
    static_classes: List[str] = []
    dynamic_classes: List[str] = []
    last_updated: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DatasetsStatusResponse(BaseModel):
    datasets: Dict[str, DatasetStatus]
    total_samples: int = 0


class RecognitionResult(BaseModel):
    label: str
    language: str = 'ASL'
    confidence: float
    sample_type: SampleTypeEnum = SampleTypeEnum.STATIC
    is_custom: bool = False
    motion_energy: float = 0.0
    is_fallback: bool = False
    description: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class RecognitionEvent(BaseModel):
    event: EventTypeEnum
    label: str
    language: Union[str, SignLanguageEnum] = SignLanguageEnum.ASL
    confidence: float = 1.0
    sample_type: SampleTypeEnum = SampleTypeEnum.STATIC
    is_custom: bool = False
    transcript: Optional[str] = None
    recent_words: Optional[List[str]] = None
    help_active: bool = False
    help_duration_seconds: float = 0.0
    timestamp: float = Field(default_factory=time.time)


class RecognitionPrediction(BaseModel):
    sign: str
    confidence: float
    mode: str  # 'STATIC', 'DYNAMIC', 'HELP', 'GEMINI_FALLBACK'
    motion_energy: float = 0.0
    is_custom: bool = False
    is_fallback: bool = False
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class RecognizeFrameRequest(BaseModel):
    image_base64: Optional[str] = None
    landmarks: Optional[List[float]] = None
    language: SignLanguageEnum = SignLanguageEnum.ASL
    include_custom: bool = True
    user_id: str = 'default_user'


class RecognizeSequenceRequest(BaseModel):
    frames: List[List[float]]
    language: SignLanguageEnum = SignLanguageEnum.ASL
    include_custom: bool = True
    user_id: str = 'default_user'


class RecognizeResponse(BaseModel):
    prediction: RecognitionPrediction
    detected_hands: List[str] = []
    debounced_sign: Optional[str] = None
    help_alert: bool = False
    gemini_reasoning: Optional[str] = None


class PolishSentenceRequest(BaseModel):
    glosses: List[str]
    context: Optional[str] = 'Daily conversation'
    target_language: str = 'English'


class PolishSentenceResponse(BaseModel):
    original_glosses: List[str]
    polished_sentence: str
    is_gemini_generated: bool
    latency_ms: float = 0.0


class DescribeSignRequest(BaseModel):
    label: str
    language: SignLanguageEnum = SignLanguageEnum.ASL
    sample_type: SampleTypeEnum = SampleTypeEnum.STATIC
    landmarks: Optional[List[float]] = None
    sequence: Optional[List[List[float]]] = None


class DescribeSignResponse(BaseModel):
    label: str
    description: str
    is_gemini_generated: bool


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: Optional[str] = 'default'
    rate: Optional[float] = 1.0


class TTSResponse(BaseModel):
    text: str
    audio_base64: Optional[str] = None
    synthesized: bool


class IngestDatasetRequest(BaseModel):
    language: SignLanguageEnum
    source_type: str = 'csv'
    source_path: Optional[str] = None
    overwrite: bool = False


class IngestDatasetResponse(BaseModel):
    success: bool
    language: SignLanguageEnum
    samples_imported: int = 0
    labels_imported: int = 0
    message: str = ''
    warnings: List[str] = []


# Aliases
CustomSignAddSamplesRequest = AddSamplesRequest
SentencePolishRequest = PolishSentenceRequest
SentencePolishResponse = PolishSentenceResponse


class RecognitionRequest(BaseModel):
    language: SignLanguageEnum = SignLanguageEnum.ASL
    features: Optional[List[float]] = None
    sequence: Optional[List[List[float]]] = None
    user_id: Optional[str] = 'default_user'
    include_custom: bool = True