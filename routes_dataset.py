"""
src/api/routes_dataset.py
-------------------------
API endpoints for dataset status, metrics, inspection, and ingestion.
"""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException, Query
from pathlib import Path

from src.ingestion.dataset_manager import get_dataset_manager
from src.ingestion.synthetic_data import generate_synthetic_dataset
from src.models.schemas import (
    DatasetStatus,
    DatasetsStatusResponse,
    IngestDatasetRequest,
    IngestDatasetResponse,
    SignLanguageEnum,
)

router = APIRouter(prefix="/api/datasets", tags=["Datasets"])


@router.get("/status", response_model=DatasetsStatusResponse)
async def get_all_datasets_status() -> DatasetsStatusResponse:
    manager = get_dataset_manager()
    return manager.get_all_statuses()


@router.get("/{language}/status", response_model=DatasetStatus)
async def get_single_dataset_status(language: SignLanguageEnum) -> DatasetStatus:
    manager = get_dataset_manager()
    try:
        return manager.get_dataset_status(language)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate-synthetic", response_model=IngestDatasetResponse)
async def generate_synthetic_data(
    language: SignLanguageEnum = Query(..., description="Target language: ASL, ISL, or BSL"),
    samples_per_class: int = Query(default=10, ge=1, le=50),
    overwrite: bool = Query(default=False),
) -> IngestDatasetResponse:
    if language == SignLanguageEnum.CUSTOM:
        raise HTTPException(status_code=400, detail="Cannot generate synthetic data for CUSTOM.")

    try:
        res = generate_synthetic_dataset(
            language=language,
            samples_per_class=samples_per_class,
            overwrite=overwrite,
        )
        return IngestDatasetResponse(
            success=True,
            language=language,
            samples_imported=res["static_imported"] + res["dynamic_imported"],
            labels_imported=res["total_labels"],
            message=f"Populated {language.value} dataset with {res['static_imported']} static and {res['dynamic_imported']} dynamic samples.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthetic generation failed: {str(e)}")


@router.post("/ingest", response_model=IngestDatasetResponse)
async def ingest_dataset_source(req: IngestDatasetRequest) -> IngestDatasetResponse:
    manager = get_dataset_manager()
    if req.language == SignLanguageEnum.ASL:
        adapter = manager.asl_adapter
    elif req.language == SignLanguageEnum.ISL:
        adapter = manager.isl_adapter
    elif req.language == SignLanguageEnum.BSL:
        adapter = manager.bsl_adapter
    else:
        raise HTTPException(status_code=400, detail="CUSTOM signs are ingested via /api/custom-sign routes.")

    if not req.source_path or not Path(req.source_path).exists():
        raise HTTPException(status_code=404, detail=f"Source file not found: {req.source_path}")

    path = Path(req.source_path)
    if req.source_type == "csv" or path.suffix.lower() == ".csv":
        ingest_res = adapter.ingest_from_csv(path)
        if not ingest_res["success"]:
            raise HTTPException(status_code=400, detail=ingest_res.get("error", "CSV ingestion failed"))
        return IngestDatasetResponse(
            success=True,
            language=req.language,
            samples_imported=ingest_res.get("total_imported", 0),
            labels_imported=len(adapter.get_status().get("labels", [])),
            message=f"Imported {ingest_res.get('total_imported', 0)} samples into {req.language.value}.",
            warnings=ingest_res.get("warnings", []),
        )
    elif req.source_type == "json" or path.suffix.lower() == ".json":
        if req.language == SignLanguageEnum.ASL:
            ingest_res = adapter.ingest_wlasl_json(path)
        elif req.language == SignLanguageEnum.ISL:
            ingest_res = adapter.ingest_include_metadata_json(path)
        elif req.language == SignLanguageEnum.BSL:
            ingest_res = adapter.ingest_bsl1k_annotations_json(path)
        else:
            ingest_res = {"success": False, "error": "Unsupported"}

        if not ingest_res["success"]:
            raise HTTPException(status_code=400, detail=ingest_res.get("error", "JSON ingestion failed"))

        return IngestDatasetResponse(
            success=True,
            language=req.language,
            samples_imported=ingest_res.get("imported", 0),
            labels_imported=len(adapter.get_status().get("labels", [])),
            message=f"Imported {ingest_res.get('imported', 0)} samples into {req.language.value}.",
            warnings=ingest_res.get("warnings", []),
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported source format: {req.source_type}")