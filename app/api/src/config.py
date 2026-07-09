from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    supabase_url: str = ""
    supabase_service_key: str = ""
    supabase_anon_key: str = ""
    supabase_storage_bucket: str = "images"

    inference_mode: Literal["local", "modal"] = "local"
    modal_app_name: str = "cannabis-maturity-inference"

    detection_model_path: str = "../../checkpoints/trichome_detection/yolov9_best.pt"
    classification_model_path: str = "../../checkpoints/trichome_classification/yolov8/medium_fold0.pt"
    segmentation_model_path: str = "../../checkpoints/stigma_segmentation/yolov8s_best.pt"

    log_level: str = "INFO"

    trichome_patch_size: int = 512
    trichome_overlap: float = 0.2

    debug_save_results: bool = False
    debug_output_dir: str = "inference_samples"


settings = Settings()
