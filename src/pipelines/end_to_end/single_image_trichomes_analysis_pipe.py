import os
import logging

from src.pipelines.end_to_end.end_to_end_utils import (
    load_obj_detection_model,
    save_visuals,
    save_results,
    compute_class_distribution,
    compute_normalized_class_distribution,
    save_class_distribution,
)

from src.pipelines.end_to_end.end_to_end_with_classification import (
    process_image,
    load_classification_model,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Detection model configuration
detection_model_config = {
    "model_name": "faster_rcnn_R_50_C4_1x",
    "checkpoint": "/home/etaylor/code_projects/thesis/checkpoints/trichomes_detection/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/29-04-2024_16-09-41/model_final.pth",
    "yaml_file": "/home/etaylor/code_projects/thesis/checkpoints/trichomes_detection/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/29-04-2024_16-09-41/config.yaml",
}

classification_models_config = {
    "trichome_classification": {
        "checkpoint": "/home/etaylor/code_projects/thesis/checkpoints/trichome_image_classification/yolo/fine_tuned/YOLOv8/Medium_dataset_0.pt"
    },
    "blur_classification": {
        "checkpoint": "/home/etaylor/code_projects/thesis/checkpoints/blur_image_classification/yolo/fine_tuned/YOLOv8/Nano_dataset_0.pt"
    },
}

# Other parameters
patch_size = 512
model_type = "yolo"
perform_blur_classification_flag = False

# Paths for the single image and output directory
image_path = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_1/images/day_9_2024_06_27/greenhouse/243/IMG_8720.JPG"
output_dir = "/home/etaylor/code_projects/thesis/src/pipelines/output_results"

# --- Model Loading ---
logger.info("Loading detection model...")
detection_model = load_obj_detection_model(detection_model_config, patch_size)

logger.info("Loading classification models...")
trichome_classification_model = load_classification_model(
    classification_models_config["trichome_classification"], model_type
)
blur_classification_model = load_classification_model(
    classification_models_config["blur_classification"], model_type
)

classification_models = {
    "trichome_classification": trichome_classification_model,
    "blur_classification": blur_classification_model,
}

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# --- Process Single Image ---
logger.info(f"Processing image: {image_path}")
results = process_image(
    image_path,
    detection_model,
    classification_models,
    model_type,
    patch_size,
    perform_blur_classification_flag=perform_blur_classification_flag,
)

# Create a directory for the image results
base_file_name = os.path.splitext(os.path.basename(image_path))[0]
image_output_dir = os.path.join(output_dir, base_file_name)
os.makedirs(image_output_dir, exist_ok=True)

# Save the visuals and result files
save_visuals(results, image_output_dir, base_file_name)
save_results(results, image_output_dir, base_file_name)

# Compute and save class distributions
class_distribution = compute_class_distribution(results.object_prediction_list)
normalized_class_distribution = compute_normalized_class_distribution(
    class_distribution
)
distribution_json_path = os.path.join(
    image_output_dir, f"{base_file_name}_class_distribution.json"
)
save_class_distribution(
    class_distribution, normalized_class_distribution, distribution_json_path
)

logger.info(f"Processing complete. Results saved to: {image_output_dir}")
