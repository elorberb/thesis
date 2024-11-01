import os
import logging
import cv2
from src.pipelines.end_to_end.end_to_end_utils import (
    load_obj_detection_model,
    perform_object_detection,
    filter_large_objects,
    extend_bounding_box,
    crop_image,
    save_visuals,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_detection_results(image_path, predictions, image_identifier_folder, margin=0.25):
    """
    Save the detected objects into images in the specified results path.

    Args:
        image_path (str): Path to the input image.
        predictions (list): List of filtered predictions.
        image_identifier_folder (str): Directory to save the result images.
        margin (float): Margin to extend the bounding boxes.
    """
    
    # get image identifier
    image_identifier = os.path.splitext(os.path.basename(image_path))[0]

    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to read image from {image_path}")
        return

    image_height, image_width = image.shape[:2]

    for idx, prediction in enumerate(predictions):
        bbox = prediction.bbox
        # Get original bounding box coordinates
        x_min = int(bbox.minx)
        y_min = int(bbox.miny)
        x_max = int(bbox.maxx)
        y_max = int(bbox.maxy)

        # Extend the bounding box
        x_min_ext, y_min_ext, x_max_ext, y_max_ext = extend_bounding_box(
            x_min, y_min, x_max, y_max, image_width, image_height, margin
        )

        # Crop the image
        cropped_img = crop_image(image, x_min_ext, y_min_ext, x_max_ext, y_max_ext)

        # Get the class name and create the class folder if it doesn't exist
        class_id = prediction.category.id
        class_mapping = {1: "clear", 2: "cloudy", 3: "amber"}
        class_name = class_mapping.get(class_id, "unknown")
        class_folder = os.path.join(image_identifier_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)

        # Save the cropped image with the bounding box information in the filename
        result_filename = f"{image_identifier}_trichome_{idx}_bbox_{x_min_ext}_{y_min_ext}_{x_max_ext}_{y_max_ext}.jpg"
        result_path = os.path.join(class_folder, result_filename)
        cv2.imwrite(result_path, cropped_img)
        logger.info(f"Saved detected object to {result_path}")


def extract_trichomes_from_image(image_path, detection_model, patch_size=512, save_results_path="results"):
    """
    Extract trichomes from the input image using the provided object detection model.

    Args:
        image_path (str): Path to the input image.
        detection_model (detectron2.modeling.meta_arch.build_model): Object detection model.
        patch_size (int): Patch size for object detection.
        save_results_path (str): Directory to save the result images.

    Returns:
        None
    """

    # Ensure the save directory exists
    os.makedirs(save_results_path, exist_ok=True)

    # Get the image identifier from the image path
    image_identifier = os.path.splitext(os.path.basename(image_path))[0]

    # Create folder for the image identifier
    image_identifier_folder = os.path.join(save_results_path, image_identifier)
    os.makedirs(image_identifier_folder, exist_ok=True)

    # Perform object detection
    detection_result = perform_object_detection(image_path, detection_model, patch_size)

    # Filter large objects
    filtered_predictions = filter_large_objects(detection_result.object_prediction_list)

    # Save the detection results
    save_detection_results(image_path, filtered_predictions, image_identifier_folder)

    # Save the full image results using save_visuals function
    save_visuals(detection_result, image_identifier_folder, image_identifier)
    

def main():
    # Set the paths and configurations
    detection_model_config = {
        "model_name": "faster_rcnn_R_50_C4_1x",
        "checkpoint": "/home/etaylor/code_projects/thesis/checkpoints/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/29-04-2024_16-09-41/model_final.pth",
        "yaml_file": "/home/etaylor/code_projects/thesis/checkpoints/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/29-04-2024_16-09-41/config.yaml",
    }
    image_path = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/images/day_9_2024_06_27/greenhouse/264/IMG_8755.JPG"
    save_results_path = "/home/etaylor/code_projects/thesis/src/pipelines/end_to_end/results/extracted_trichomes_images"
    patch_size = 512  # Adjust if necessary
    
    # Load the object detection model
    detection_model = load_obj_detection_model(detection_model_config, patch_size)

    # Extract trichomes from the input image
    extract_trichomes_from_image(image_path, detection_model, patch_size, save_results_path)


if __name__ == "__main__":
    main()
