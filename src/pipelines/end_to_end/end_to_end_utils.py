import time
import os
import logging
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import re
import json
from collections import Counter
import numpy as np
from PIL import ImageDraw
import warnings

warnings.filterwarnings(action="ignore")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_obj_detection_model(model_config, patch_size=512):
    logger.info("Loading the model.")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="detectron2",
        model_path=model_config["checkpoint"],
        config_path=model_config["yaml_file"],
        confidence_threshold=0.5,
        image_size=patch_size,
        device="cuda:0",  # or 'cpu'
    )

    return detection_model


def perform_trichome_detection(image_path, detection_model, patch_size=512):
    logger.info(f"Performing object detection on image: {os.path.basename(image_path)}")
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
    detection_time = time.time() - start_time
    logger.info(
        f"Time taken for object detection on image {os.path.basename(image_path)}: {detection_time:.2f} seconds"
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


def extend_bounding_box(
    x_min, y_min, x_max, y_max, image_width, image_height, margin=0.25
):
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    x_min_extended = max(0, x_min - int(margin * bbox_width))
    y_min_extended = max(0, y_min - int(margin * bbox_height))
    x_max_extended = min(image_width, x_max + int(margin * bbox_width))
    y_max_extended = min(image_height, y_max + int(margin * bbox_height))

    return x_min_extended, y_min_extended, x_max_extended, y_max_extended


def crop_image(image, x_min, y_min, x_max, y_max):
    return image[y_min:y_max, x_min:x_max]


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


def save_visuals(result, image_output_dir, base_file_name, box_color=(0, 255, 0)):
    """
    Exports visualization with bounding boxes in the specified color.

    Parameters:
        result: The prediction result object from SAHI.
        image_output_dir (str): Directory to save the output image.
        base_file_name (str): Base name for the saved image file.
        box_color (tuple): BGR color tuple for bounding boxes (default is green: (0, 255, 0)).
    """
    logger.info(f"Exporting visuals for {base_file_name}")
    start_time = time.time()

    # Convert box_color from BGR to RGB as SAHI expects RGB
    # box_color_rgb = box_color[::-1]

    result.export_visuals(
        export_dir=image_output_dir,
        text_size=1,
        rect_th=2,
        hide_labels=True,
        hide_conf=True,
        file_name=base_file_name
    )

    export_time = time.time() - start_time
    logger.info(f"Time taken to export visuals: {export_time:.2f} seconds")


def save_visuals_single_color(result, image_output_dir, base_file_name, box_color=(0, 255, 0)):
    """
    Draws all detection boxes on the PIL image from results in the same color and saves it.

    Parameters:
        result: SAHI result object containing `image` (PIL Image) and `object_prediction_list`.
        image_output_dir (str): Directory to save the output image.
        base_file_name (str): Base name used for saving the output image.
        box_color (tuple): RGB color tuple for all bounding boxes (default is green).
    """
    logger.info(f"Exporting visuals for {base_file_name} with uniform colored boxes.")
    start_time = time.time()

    # Get the PIL image directly from result
    image = result.image.copy()

    # Initialize drawing context
    draw = ImageDraw.Draw(image)

    # Iterate over predictions and draw bounding boxes
    for prediction in result.object_prediction_list:
        x_min = int(prediction.bbox.minx)
        y_min = int(prediction.bbox.miny)
        x_max = int(prediction.bbox.maxx)
        y_max = int(prediction.bbox.maxy)

        # Draw rectangle on the image (outline only)
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=box_color, width=2)

    # Ensure output directory exists
    os.makedirs(image_output_dir, exist_ok=True)

    # Save the image
    output_path = os.path.join(image_output_dir, f"{base_file_name}_visuals.jpg")
    image.save(output_path)

    export_time = time.time() - start_time
    logger.info(f"Time taken to export visuals: {export_time:.2f} seconds")


def save_results(result, image_output_dir, base_file_name):
    logger.info(f"Saving results for {base_file_name}")
    # Save results to JSON with 'raw' suffix
    json_path = os.path.join(image_output_dir, f"{base_file_name}_raw.json")
    with open(json_path, "w") as json_file:
        json.dump(result.to_coco_predictions(), json_file)
    logger.info(f"Results saved to JSON: {json_path}")


def non_max_suppression(predictions, iou_threshold=0.7):
    if len(predictions) == 0:
        return predictions

    # Sort predictions by confidence score in descending order
    predictions = sorted(predictions, key=lambda x: x.score.value, reverse=True)

    keep = []
    while len(predictions) > 0:
        # Pick the highest confidence detection and remove it from the list
        highest = predictions.pop(0)
        keep.append(highest)

        # Compare IoU with remaining predictions
        predictions = [
            pred
            for pred in predictions
            if compute_iou(highest.bbox, pred.bbox) < iou_threshold
        ]

    return keep


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


def compute_iou(box1, box2):
    # Calculate intersection area
    x1 = max(box1.minx, box2.minx)
    y1 = max(box1.miny, box2.miny)
    x2 = min(box1.maxx, box2.maxx)
    y2 = min(box1.maxy, box2.maxy)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate areas
    area1 = (box1.maxx - box1.minx) * (box1.maxy - box1.miny)
    area2 = (box2.maxx - box2.minx) * (box2.maxy - box2.miny)

    # Calculate IoU
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0
