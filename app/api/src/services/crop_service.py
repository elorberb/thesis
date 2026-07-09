from __future__ import annotations

import cv2
import httpx
import numpy as np
from cannabis_maturity.crop_extractor import CropExtractor
from cannabis_maturity.models import StigmaResult, TrichomeResult


class CropService:
    _TIMEOUT_SECONDS = 30.0

    @staticmethod
    def regenerate_crops(
        image_url: str, detections: dict
    ) -> tuple[list[str] | None, list[str] | None]:
        trichome_detections = detections.get("trichomes") or []
        stigma_detections = detections.get("stigmas") or []
        if not trichome_detections and not stigma_detections:
            return None, None

        image_bgr = CropService._download_image(image_url)
        if image_bgr is None:
            return None, None

        trichome_crops: list[str] | None = None
        stigma_crops: list[str] | None = None

        if trichome_detections:
            trichome_result = TrichomeResult(
                detections=trichome_detections,
                distribution={},
                total_count=len(trichome_detections),
            )
            trichome_crops = CropExtractor.extract_trichome_crops(image_bgr, trichome_result)

        if stigma_detections:
            stigma_result = StigmaResult(
                detections=stigma_detections,
                avg_green_ratio=0.0,
                avg_orange_ratio=0.0,
                total_count=len(stigma_detections),
            )
            stigma_crops = CropExtractor.extract_stigma_crops(image_bgr, stigma_result)

        return trichome_crops, stigma_crops

    @staticmethod
    def _download_image(image_url: str) -> np.ndarray | None:
        try:
            response = httpx.get(image_url, timeout=CropService._TIMEOUT_SECONDS)
            response.raise_for_status()
        except httpx.HTTPError:
            return None
        image_array = np.frombuffer(response.content, np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
