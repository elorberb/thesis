import time
import os
import logging
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import re
import json
from collections import Counter
import numpy as np

import warnings

from src.pipelines.end_to_end.end_to_end_utils import load_obj_detection_model

warnings.filterwarnings(action="ignore")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_image(image_path, detection_model, patch_size=512):
    logger.info(f"Processing image: {os.path.basename(image_path)}")
    start_time = time.time()
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=patch_size,
        slice_width=patch_size,
        overlap_height_ratio=0,
        overlap_width_ratio=0,
        verbose=True,
    )
    processing_time = time.time() - start_time
    logger.info(
        f"Time taken to process image {os.path.basename(image_path)}: {processing_time:.2f} seconds"
    )
    return result


def filter_large_objects(predictions, size_threshold_ratio=10):
    # Calculate the size of each detected object
    sizes = [
        (pred.bbox.maxx - pred.bbox.minx) * (pred.bbox.maxy - pred.bbox.miny)
        for pred in predictions
    ]
    if not sizes:
        return predictions  # No detections to filter

    # Calculate the median size
    median_size = np.median(sizes)

    # Filter out objects that are larger than the threshold ratio of the median size
    filtered_predictions = [
        pred
        for pred in predictions
        if (pred.bbox.maxx - pred.bbox.minx) * (pred.bbox.maxy - pred.bbox.miny)
        <= median_size * size_threshold_ratio
    ]

    return filtered_predictions


def save_visuals(result, image_output_dir, base_file_name):
    logger.info(f"Exporting visuals for {base_file_name}")
    start_time = time.time()
    result.export_visuals(
        export_dir=image_output_dir,
        text_size=1,
        rect_th=2,
        hide_labels=True,
        hide_conf=True,
        file_name=base_file_name,
    )
    export_time = time.time() - start_time
    logger.info(f"Time taken to export visuals: {export_time:.2f} seconds")


def save_results(result, image_output_dir, base_file_name):
    logger.info(f"Saving results for {base_file_name}")
    # Save results to JSON with 'raw' suffix
    json_path = os.path.join(image_output_dir, f"{base_file_name}_raw.json")
    with open(json_path, "w") as json_file:
        json.dump(result.to_coco_predictions(), json_file)
    logger.info(f"Results saved to JSON: {json_path}")


def compute_class_distribution(predictions):
    class_counts = Counter()
    for prediction in predictions:
        class_label = prediction.category.id
        class_counts[class_label] += 1
    return class_counts


def compute_normalized_class_distribution(class_counts):
    total_count = sum(class_counts.values())
    normalized_distribution = {k: v / total_count for k, v in class_counts.items()}
    return normalized_distribution


def save_class_distribution(
    class_distribution, normalized_class_distribution, output_path
):
    combined_distribution = {
        "class_distribution": class_distribution,
        "normalized_class_distribution": normalized_class_distribution,
    }
    with open(output_path, "w") as json_file:
        json.dump(combined_distribution, json_file)
    logger.info(
        f"Class and normalized class distributions saved to JSON: {output_path}"
    )


def process_images_in_folder(folder_path, detection_model, output_dir, patch_size):
    logger.info(f"Processing images in folder: {os.path.basename(folder_path)}")

    # Extract the folder number
    folder_number = os.path.basename(folder_path)

    # Create the directory for saving results
    result_dir = os.path.join(output_dir, folder_number)
    os.makedirs(result_dir, exist_ok=True)

    # Regex to extract numerical part from filenames like IMG_4915.JPG
    def extract_number(filename):
        match = re.search(r"\d+", filename)
        return int(match.group()) if match else -1

    # Get all image files in the folder
    image_files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Sort files based on the numerical part
    image_files.sort(key=extract_number)

    aggregated_results = []

    # Start timing the processing
    start_time = time.time()

    # Skip the first image
    for image_file in image_files[1:]:
        image_path = os.path.join(folder_path, image_file)
        base_file_name = os.path.splitext(image_file)[0]

        # Create a directory for each image inside the result folder
        image_output_dir = os.path.join(result_dir, base_file_name)
        os.makedirs(image_output_dir, exist_ok=True)

        result = process_image(image_path, detection_model, patch_size)

        # Filter out large objects
        filtered_predictions = filter_large_objects(result.object_prediction_list)

        # Update result with filtered predictions
        result.object_prediction_list = filtered_predictions

        # Export results with filtered predictions
        save_visuals(result, image_output_dir, base_file_name)
        save_results(result, image_output_dir, base_file_name)
        # Compute class label distribution and normalized distribution for the image
        class_distribution = compute_class_distribution(filtered_predictions)
        normalized_class_distribution = compute_normalized_class_distribution(
            class_distribution
        )

        # Save combined distributions to JSON
        distribution_json_path = os.path.join(
            image_output_dir, f"{base_file_name}_class_distribution.json"
        )
        save_class_distribution(
            class_distribution, normalized_class_distribution, distribution_json_path
        )

        aggregated_results.extend(filtered_predictions)

    # Calculate total processing time
    total_time = time.time() - start_time
    logger.info(f"Total time taken to process the folder: {total_time:.2f} seconds")

    # Compute and save aggregated class label distribution for all images
    compute_aggregated_class_distribution(aggregated_results, result_dir)


def compute_aggregated_class_distribution(aggregated_results, output_dir):
    logger.info("Computing aggregated class label distribution.")
    aggregated_class_counts = Counter()
    for prediction in aggregated_results:
        class_label = prediction.category.id
        aggregated_class_counts[class_label] += 1

    # Compute normalized aggregated class label distribution
    normalized_aggregated_class_distribution = compute_normalized_class_distribution(
        aggregated_class_counts
    )

    # Save combined distributions to JSON
    aggregated_class_distribution_json_path = os.path.join(
        output_dir, "class_distribution.json"
    )
    save_class_distribution(
        aggregated_class_counts,
        normalized_aggregated_class_distribution,
        aggregated_class_distribution_json_path,
    )


def process_all_folders(
    parent_folder_path, detection_model, output_base_dir, patch_size
):

    start_time = time.time()  # Start timer
    # Get all subfolders in the parent folder
    subfolders = [f.path for f in os.scandir(parent_folder_path) if f.is_dir()]

    for folder_path in subfolders:
        logger.info(f"Processing folder: {folder_path}")
        process_images_in_folder(
            folder_path, detection_model, output_base_dir, patch_size
        )

    total_time = time.time() - start_time  # End timer
    logger.info(f"Total time taken to process all folders: {total_time:.2f} seconds")


def main():
    model_config = {
        "model_name": "faster_rcnn_R_50_C4_1x",
        "checkpoint": "/home/etaylor/code_projects/thesis/checkpoints/trichomes_detection/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/29-04-2024_16-09-41/model_final.pth",
        "yaml_file": "/home/etaylor/code_projects/thesis/checkpoints/trichomes_detection/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/29-04-2024_16-09-41/config.yaml",
    }

    parent_input_folder = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_2/images/day_9_2025_01_16/lab"
    output_base_folder = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_2/results/faster_rcnn/day_9_2025_01_16/lab"

    # Ensure output base directory exists
    os.makedirs(output_base_folder, exist_ok=True)

    patch_size = 512

    detection_model = load_obj_detection_model(model_config, patch_size)
    process_all_folders(
        parent_input_folder, detection_model, output_base_folder, patch_size
    )


if __name__ == "__main__":
    main()
