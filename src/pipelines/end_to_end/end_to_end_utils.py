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


def perform_object_detection(image_path, detection_model, patch_size=512):
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
    # Calculate sizes of bounding boxes
    sizes = [
        (pred.bbox.maxx - pred.bbox.minx) * (pred.bbox.maxy - pred.bbox.miny)
        for pred in predictions
    ]

    if sizes:
        # Calculate median size
        median_size = np.median(sizes)

        # Filter predictions based on size threshold
        filtered_predictions = [
            pred
            for pred in predictions
            if (pred.bbox.maxx - pred.bbox.minx) * (pred.bbox.maxy - pred.bbox.miny)
            <= median_size * size_threshold_ratio
        ]

        # Log information
        logger.info(
            f"Filtered {len(predictions) - len(filtered_predictions)} large objects"
        )

        return filtered_predictions

    return predictions


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
