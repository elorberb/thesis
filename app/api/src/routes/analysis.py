from __future__ import annotations

import base64
from datetime import datetime

from cannabis_maturity.maturity_assessor import MaturityAssessor
from cannabis_maturity.models import TrichomeType
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from supabase import create_client

from auth import get_current_user
from config import settings
from models.schemas import (
    AnalysisListItem,
    AnalysisListResponse,
    AnalysisPatch,
    AnalyzeResponse,
    CorrectionsUpdate,
    PlantAnalysisHistory,
    PlantAnalysisItem,
    PlantCreate,
    PlantListResponse,
    PlantResponse,
)
from services.crop_service import CropService
from services.database_service import DatabaseService
from services.inference_error import InferenceError
from services.storage_service import StorageService

router = APIRouter(tags=["analysis"])

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
_MAX_IMAGE_BYTES = 20 * 1024 * 1024


def _supabase():
    return create_client(settings.supabase_url, settings.supabase_service_key)


def _analysis_to_response(
    record: dict,
    trichome_crops_b64: list[str] | None = None,
    stigma_crops_b64: list[str] | None = None,
) -> AnalyzeResponse:
    detections = record.get("detections") or {}
    trichome_dist = record.get("trichome_distribution") or {"clear": 0, "cloudy": 0, "amber": 0}
    stigma_ratios = record.get("stigma_ratios") or {"green": 0.0, "orange": 0.0}
    return AnalyzeResponse(
        id=record["id"],
        created_at=datetime.fromisoformat(record["created_at"]),
        user_id=record["performed_by"],
        plant_id=record.get("plant_id"),
        image_url=record["image_url"],
        annotated_image_url=record.get("annotated_image_url"),
        trichome_result={
            "detections": detections.get("trichomes") or [],
            "distribution": trichome_dist,
            "total_count": sum(trichome_dist.values()),
        },
        stigma_result={
            "detections": detections.get("stigmas") or [],
            "avg_green_ratio": stigma_ratios.get("green", 0.0),
            "avg_orange_ratio": stigma_ratios.get("orange", 0.0),
            "total_count": len(detections.get("stigmas") or []),
        },
        maturity_stage=record["maturity_stage"],
        recommendation=record["recommendation"],
        trichome_crops_b64=trichome_crops_b64,
        stigma_crops_b64=stigma_crops_b64,
    )


# ─── ANALYZE ─────────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse, status_code=status.HTTP_201_CREATED)
async def analyze_image(
    request: Request,
    file: UploadFile,
    plant_id: str | None = None,
    user_id: str = Depends(get_current_user),
) -> AnalyzeResponse:
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported image type: {file.content_type}",
        )

    image_bytes = await file.read()
    if len(image_bytes) > _MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image exceeds 20 MB limit",
        )

    storage = StorageService(_supabase(), settings.supabase_storage_bucket)
    db = DatabaseService(_supabase())

    image_url = storage.upload_image(image_bytes, content_type=file.content_type or "image/jpeg")

    try:
        result: dict = await request.app.state.inference_service.analyze(image_bytes)
    except InferenceError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    annotated_image_url: str | None = None
    if result.get("annotated_image_b64"):
        annotated_bytes = base64.b64decode(result["annotated_image_b64"])
        annotated_image_url = storage.upload_image(annotated_bytes, content_type="image/jpeg")

    record = db.save_analysis(user_id, image_url, annotated_image_url, result, plant_id)

    return AnalyzeResponse(
        id=record["id"],
        created_at=datetime.fromisoformat(record["created_at"]),
        user_id=user_id,
        plant_id=plant_id,
        image_url=image_url,
        annotated_image_url=annotated_image_url,
        trichome_result=result["trichome_result"],
        stigma_result=result["stigma_result"],
        maturity_stage=result["maturity_stage"],
        recommendation=result["recommendation"],
        trichome_crops_b64=result.get("trichome_crops_b64"),
        stigma_crops_b64=result.get("stigma_crops_b64"),
    )


# ─── ANALYSES LIST / GET / DELETE / PATCH ────────────────────────────────────

@router.get("/analyses", response_model=AnalysisListResponse)
def list_analyses(
    limit: int = 20,
    user_id: str = Depends(get_current_user),
) -> AnalysisListResponse:
    db = DatabaseService(_supabase())
    records = db.list_analyses(user_id=user_id, limit=limit)
    items = [
        AnalysisListItem(
            id=r["id"],
            created_at=datetime.fromisoformat(r["created_at"]),
            user_id=r["performed_by"],
            plant_id=r.get("plant_id"),
            image_url=r["image_url"],
            annotated_image_url=r.get("annotated_image_url"),
            maturity_stage=r["maturity_stage"],
            recommendation=r["recommendation"],
            trichome_distribution=r.get("trichome_distribution"),
            stigma_ratios=r.get("stigma_ratios"),
        )
        for r in records
    ]
    return AnalysisListResponse(items=items, total=len(items))


@router.get("/analyses/{analysis_id}", response_model=AnalyzeResponse)
def get_analysis(
    analysis_id: str,
    user_id: str = Depends(get_current_user),
) -> AnalyzeResponse:
    db = DatabaseService(_supabase())
    record = db.get_analysis(analysis_id, user_id=user_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    trichome_crops, stigma_crops = CropService.regenerate_crops(
        record["image_url"], record.get("detections") or {}
    )
    return _analysis_to_response(record, trichome_crops, stigma_crops)


@router.patch("/analyses/{analysis_id}", response_model=AnalyzeResponse)
def patch_analysis(
    analysis_id: str,
    body: AnalysisPatch,
    user_id: str = Depends(get_current_user),
) -> AnalyzeResponse:
    db = DatabaseService(_supabase())
    record = db.get_analysis(analysis_id, user_id=user_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    fields = body.model_dump(mode="json", exclude_none=True)
    if not fields:
        return _analysis_to_response(record)
    updated = db.update_analysis(analysis_id, fields)
    return _analysis_to_response(updated)


@router.patch("/analyses/{analysis_id}/corrections", response_model=AnalyzeResponse)
def save_corrections(
    analysis_id: str,
    body: CorrectionsUpdate,
    user_id: str = Depends(get_current_user),
) -> AnalyzeResponse:
    db = DatabaseService(_supabase())
    record = db.get_analysis(analysis_id, user_id=user_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    trichome_dist: dict = body.trichome_distribution or record.get("trichome_distribution") or {}
    stigma_ratios: dict = body.stigma_ratios or record.get("stigma_ratios") or {}

    typed_dist: dict[TrichomeType, int] = {
        TrichomeType(k): int(v) for k, v in trichome_dist.items()
    }
    maturity_stage, recommendation = MaturityAssessor.assess(
        distribution=typed_dist,
        avg_green_ratio=float(stigma_ratios.get("green", 0.0)),
        avg_orange_ratio=float(stigma_ratios.get("orange", 0.0)),
    )

    corrections = body.model_dump(exclude_none=True)
    updated = db.save_corrections(
        analysis_id=analysis_id,
        corrections=corrections,
        trichome_distribution=trichome_dist,
        stigma_ratios=stigma_ratios,
        maturity_stage=maturity_stage.value,
        recommendation=recommendation,
        detections=body.detections,
    )
    return _analysis_to_response(updated)


@router.delete("/analyses/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_analysis(
    analysis_id: str,
    user_id: str = Depends(get_current_user),
) -> None:
    db = DatabaseService(_supabase())
    if not db.get_analysis(analysis_id, user_id=user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    db.delete_analysis(analysis_id)


# ─── PLANTS ──────────────────────────────────────────────────────────────────

@router.post("/plants", response_model=PlantResponse, status_code=status.HTTP_201_CREATED)
def create_plant(
    body: PlantCreate,
    user_id: str = Depends(get_current_user),
) -> PlantResponse:
    db = DatabaseService(_supabase())
    record = db.create_plant(name=body.name, metadata=body.metadata, created_by=user_id)
    return PlantResponse(
        id=record["id"],
        created_by=record.get("created_by"),
        name=record["name"],
        status=record["status"],
        metadata=record.get("metadata") or {},
        created_at=datetime.fromisoformat(record["created_at"]),
    )


@router.get("/plants", response_model=PlantListResponse)
def list_plants(
    name: str | None = None,
    user_id: str = Depends(get_current_user),
) -> PlantListResponse:
    db = DatabaseService(_supabase())
    records = db.list_plants(user_id=user_id, name=name)
    items = [
        PlantResponse(
            id=r["id"],
            created_by=r.get("created_by"),
            name=r["name"],
            status=r["status"],
            metadata=r.get("metadata") or {},
            created_at=datetime.fromisoformat(r["created_at"]),
        )
        for r in records
    ]
    return PlantListResponse(items=items, total=len(items))


@router.delete("/plants/{plant_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_plant(
    plant_id: str,
    user_id: str = Depends(get_current_user),
) -> None:
    db = DatabaseService(_supabase())
    if not db.get_plant(plant_id, user_id=user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plant not found")
    db.delete_plant(plant_id)


# ─── LINK ANALYSIS TO PLANT ──────────────────────────────────────────────────

@router.patch("/analyses/{analysis_id}/plant")
def link_analysis_to_plant(
    analysis_id: str,
    plant_id: str,
    user_id: str = Depends(get_current_user),
) -> dict:
    db = DatabaseService(_supabase())
    if not db.get_analysis(analysis_id, user_id=user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    db.link_plant_to_analysis(analysis_id, plant_id)
    return {"analysis_id": analysis_id, "plant_id": plant_id}


# ─── PLANT ANALYSIS HISTORY ──────────────────────────────────────────────────

@router.get("/plants/{plant_id}/analyses", response_model=PlantAnalysisHistory)
def list_plant_analyses(
    plant_id: str,
    user_id: str = Depends(get_current_user),
) -> PlantAnalysisHistory:
    db = DatabaseService(_supabase())
    records = db.list_plant_analyses(plant_id, user_id=user_id)
    items = [
        PlantAnalysisItem(
            id=r["id"],
            created_at=datetime.fromisoformat(r["created_at"]),
            user_id=r["performed_by"],
            plant_id=r.get("plant_id"),
            image_url=r["image_url"],
            annotated_image_url=r.get("annotated_image_url"),
            maturity_stage=r["maturity_stage"],
            recommendation=r["recommendation"],
            trichome_distribution=r.get("trichome_distribution"),
            stigma_ratios=r.get("stigma_ratios"),
        )
        for r in records
    ]
    return PlantAnalysisHistory(plant_id=plant_id, items=items, total=len(items))
