"""
src/api/app.py
--------------
Main FastAPI Application Entrypoint.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import ASL_DATA_DIR, BSL_DATA_DIR, ISL_DATA_DIR, STATIC_DIR
from src.ingestion.dataset_manager import get_dataset_manager
from src.ingestion.synthetic_data import generate_synthetic_dataset
from src.models.database import init_db
from src.models.schemas import SignLanguageEnum

from src.api.routes_dataset import router as dataset_router
from src.api.routes_custom import router as custom_router
from src.api.routes_recognition import router as recognition_router
from src.api.routes_gemini import router as gemini_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    manager = get_dataset_manager()
    for lang in [SignLanguageEnum.ASL, SignLanguageEnum.ISL, SignLanguageEnum.BSL]:
        status = manager.get_dataset_status(lang)
        if status.sample_count == 0:
            generate_synthetic_dataset(language=lang, samples_per_class=5, seed=42)
    yield


app = FastAPI(
    title="Multilingual & Personalized Sign Language Data System",
    description="Accessible data ingestion, landmark processing, Teach My Sign custom signs, and Meeting Mode.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dataset_router)
app.include_router(custom_router)
app.include_router(recognition_router)
app.include_router(gemini_router)

STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "js").mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Sign Language System API is running. UI index.html ready."}


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}
