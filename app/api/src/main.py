from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routes import analysis
from services.local_inference_service import LocalInferenceService
from services.modal_client import ModalClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if settings.inference_mode == "modal":
        app.state.inference_service = ModalClient(settings.modal_app_name)
    else:
        app.state.inference_service = LocalInferenceService(
            detection_model_path=settings.detection_model_path,
            classification_model_path=settings.classification_model_path,
            segmentation_model_path=settings.segmentation_model_path,
            trichome_patch_size=settings.trichome_patch_size,
            trichome_overlap=settings.trichome_overlap,
            debug_save_results=settings.debug_save_results,
            debug_output_dir=settings.debug_output_dir,
        )
    yield


app = FastAPI(
    title="Cannabis Maturity Analysis API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis.router, prefix="/api/v1")


@app.get("/health")
@app.get("/api/v1/health")
def health_check() -> dict:
    return {"status": "ok"}
