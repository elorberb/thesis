import os
import cv2
import logging
import numpy as np
from fastai.vision.all import *
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO

# Remove conflicting import
# from PIL import Image as PILImage
# If needed, import PIL.Image with a different alias
from PIL import Image as PilImage

from src.pipelines.end_to_end.end_to_end_utils import (
    load_obj_detection_model,
    perform_trichome_detection,
    filter_large_objects,
    extend_bounding_box,
    crop_image,
    non_max_suppression,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Custom Transform to Resize with Padding
def custom_transform(size):
    return Resize(size, method="pad", pad_mode="zeros")


# Custom Transform to HSV using fastai's rgb2hsv
class RGB2HSV(Transform):
    def encodes(self, img: PILImage):
        return rgb2hsv(img)


# Utility Functions


def configure_output_directory(image_path, base_output_dir):
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(base_output_dir, file_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_high_res_plot(output_path):
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# Load classification models
def load_classification_model(model_config, model_type):
    if model_type == "fastai":
        logger.info("Loading FastAI classification model.")
        return load_learner(model_config["checkpoint"])
    elif model_type == "yolo":
        logger.info("Loading YOLO classification model.")
        return YOLO(model_config["checkpoint"])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def classify_bbox_image(cropped_image, classification_model, model_type):
    if model_type == "fastai":
        cropped_pil_image = (
            PILImage.create(cropped_image)
            if isinstance(cropped_image, np.ndarray)
            else cropped_image
        )
        pred_class, _, _ = classification_model.predict(cropped_pil_image)
        return pred_class
    elif model_type == "yolo":
        results = classification_model(cropped_image)
        return int(results[0].probs.data.argmax())
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def plot_classified_bbox(
    cropped_image,
    detection_class_name,
    classification_class_name,
    output_dir,
    index,
    prefix="classified_object",
):
    # Create the classified objects folder if it doesn't exist
    classified_objects_dir = os.path.join(output_dir, "classified_objects")
    os.makedirs(classified_objects_dir, exist_ok=True)

    plt.figure()
    plt.title(
        f"Detected Class: {detection_class_name}\nPredicted Class: {classification_class_name}"
    )
    plt.imshow(cropped_image)
    plt.axis("off")

    # Save the figure in the classified objects folder
    output_path = os.path.join(classified_objects_dir, f"{prefix}_{index}.png")
    save_high_res_plot(output_path)


def classify_trichomes(
    image_path, result, classification_model, output_dir, model_type
):
    logger.info("Classifying good quality detected trichomes.")
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape
    start_classification = time.time()

    # Mapping from classification model class IDs to detection model class IDs
    classification_to_detection_id_mapping = {
        0: 3,  # 'amber' maps to 'Amber'
        1: 1,  # 'clear' maps to 'Clear'
        2: 2,  # 'cloudy' maps to 'Cloudy'
    }

    # Mapping from class IDs to class names for both models
    detection_class_id_to_name = {1: "Clear", 2: "Cloudy", 3: "Amber"}
    classification_class_id_to_name = {0: "amber", 1: "clear", 2: "cloudy"}

    for idx, prediction in enumerate(result.object_prediction_list):

        # Save the original class id for the detection model
        detection_model_pred_class_id = prediction.category.id

        # Get the original bounding box coordinates
        x_min = int(prediction.bbox.minx)
        y_min = int(prediction.bbox.miny)
        x_max = int(prediction.bbox.maxx)
        y_max = int(prediction.bbox.maxy)

        # Extend bounding box by 25%
        x_min_ext, y_min_ext, x_max_ext, y_max_ext = extend_bounding_box(
            x_min, y_min, x_max, y_max, image_width, image_height
        )

        # Crop the extended bounding box from the original image
        cropped_image = crop_image(image, x_min_ext, y_min_ext, x_max_ext, y_max_ext)

        # Convert to PIL Image
        if isinstance(cropped_image, np.ndarray):
            cropped_pil_image = PILImage.create(cropped_image)
        else:
            cropped_pil_image = cropped_image

        # Classify the cropped image using the classification model
        classification_model_pred = classify_bbox_image(
            cropped_image, classification_model, model_type
        )

        if model_type == "fastai":
            # Convert class name to class ID in the classification model
            classification_model_pred_class_name_lower = (
                classification_model_pred.lower()
            )
            classification_model_pred_class_id = list(
                classification_class_id_to_name.keys()
            )[
                list(classification_class_id_to_name.values()).index(
                    classification_model_pred_class_name_lower
                )
            ]
        else:
            classification_model_pred_class_id = classification_model_pred

        # Map classification model class ID to detection model class ID
        detection_model_class_id = classification_to_detection_id_mapping.get(
            classification_model_pred_class_id, 0
        )

        # Update the prediction category id and name
        prediction.category.id = detection_model_class_id
        prediction.category.name = detection_class_id_to_name.get(
            detection_model_class_id, "Unknown"
        )

        # Get class names for logging and plotting
        faster_rcnn_class_name = detection_class_id_to_name.get(
            detection_model_pred_class_id, "Unknown"
        )
        classification_class_name = classification_class_id_to_name.get(
            classification_model_pred_class_id, "Unknown"
        )

        # Plot the classified object
        plot_classified_bbox(
            cropped_pil_image,
            faster_rcnn_class_name,
            classification_class_name,
            output_dir,
            idx,
        )

        # logger.info(
        #     f"Detected Class (Faster R-CNN): {detection_model_pred_class_id} {faster_rcnn_class_name}"
        # )
        # logger.info(
        #     f"Predicted Class (Classification Model): {classification_model_pred_class_id} {classification_class_name}"
        # )

    end_classification = time.time() - start_classification
    # logger.info(f"Time taken for classification: {end_classification:.2f} seconds")


def perform_blur_classification(
    image_path, predictions, blur_classification_model, model_type
):
    logger.info("Filtering out blurry objects.")
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape
    filtered_predictions = []
    blurry_trichomes = []  # List to hold blurry predictions and their crops

    for prediction in predictions:
        # Bounding box extraction and extension
        x_min = int(prediction.bbox.minx)
        y_min = int(prediction.bbox.miny)
        x_max = int(prediction.bbox.maxx)
        y_max = int(prediction.bbox.maxy)

        # Extend bounding box if needed
        x_min_ext, y_min_ext, x_max_ext, y_max_ext = extend_bounding_box(
            x_min, y_min, x_max, y_max, image_width, image_height
        )

        # Crop and preprocess the image
        cropped_image = crop_image(image, x_min_ext, y_min_ext, x_max_ext, y_max_ext)

        # Classify for blur detection
        is_sharp = classify_bbox_image(
            cropped_image, blur_classification_model, model_type
        )
        # fastai format return string and yolo return int
        if is_sharp == "good_quality" or is_sharp == 1:
            filtered_predictions.append(prediction)
        else:
            blurry_trichomes.append((prediction, cropped_image))

    logger.info(
        f"Number of predictions after blur filtering: {len(filtered_predictions)}"
    )
    logger.info(
        f"Number of filtered predictions: {len(predictions) - len(filtered_predictions)}"
    )
    return filtered_predictions, blurry_trichomes


def classify_blurry_trichomes(
    blurry_trichomes, trichome_classification_model, output_dir, model_type
):

    logger.info("Classifying and plotting blurry trichomes.")
    # Mapping from classification model class IDs to class names
    classification_class_id_to_name = {0: "amber", 1: "clear", 2: "cloudy"}

    for idx, (prediction, cropped_image) in enumerate(blurry_trichomes):
        # Classify the cropped image using the classification model
        classification_model_pred = classify_bbox_image(
            cropped_image, trichome_classification_model, model_type
        )

        # Convert class name to class ID in the classification model
        if model_type == "fastai":
            classification_model_pred_class_name_lower = (
                classification_model_pred.lower()
            )
            classification_model_pred_class_id = list(
                classification_class_id_to_name.keys()
            )[
                list(classification_class_id_to_name.values()).index(
                    classification_model_pred_class_name_lower
                )
            ]
        else:
            classification_model_pred_class_id = classification_model_pred

        # Get class names for logging and plotting
        classification_class_name = classification_class_id_to_name.get(
            classification_model_pred_class_id, "Unknown"
        )

        # Plot the classified object
        plot_classified_bbox(
            cropped_image,
            "Blurry",
            classification_class_name,
            output_dir,
            idx,
            prefix="blurry_trichome",
        )

        logger.info(
            f"Blurry Trichome Predicted Class (Classification Model): {classification_model_pred_class_id} {classification_class_name}"
        )


def calc_trichome_distribution(predictions):
    logger.info("Calculating trichome distribution.")
    class_counts = {}
    total = len(predictions)

    for prediction in predictions:
        class_id = prediction.category.id
        class_counts[class_id] = class_counts.get(class_id, 0) + 1

    # Calculate percentages
    distribution = {
        class_id: (count / total) * 100 for class_id, count in class_counts.items()
    }

    # Calculate normalized distribution
    normalized_distribution = {
        class_id: count / total for class_id, count in class_counts.items()
    }

    # Correct class IDs to class names mapping
    labels = {1: "Clear", 2: "Cloudy", 3: "Amber"}
    distribution_named = {labels.get(k, "Unknown"): v for k, v in distribution.items()}
    normalized_distribution_named = {
        labels.get(k, "Unknown"): v for k, v in normalized_distribution.items()
    }

    logger.info(f"Trichome Distribution (Percentage): {distribution_named}")
    logger.info(f"Normalized Trichome Distribution: {normalized_distribution_named}")

    return {
        "percentage_distribution": distribution_named,
        "normalized_distribution": normalized_distribution_named,
    }


def plot_image(image_path, predictions, output_dir, title, filename):
    # Read the image
    image = cv2.imread(image_path)

    # Define color mapping for classes (in BGR format)
    class_color_mapping = {
        1: (128, 128, 128),  # Clear -> Grey
        2: (255, 255, 255),  # Cloudy -> White
        3: (0, 165, 255),  # Amber -> Orange (BGR format)
    }

    # Iterate over predictions
    for prediction in predictions:
        x_min = int(prediction.bbox.minx)
        y_min = int(prediction.bbox.miny)
        x_max = int(prediction.bbox.maxx)
        y_max = int(prediction.bbox.maxy)

        # Get the class name
        label = prediction.category.id

        # Get color for the class
        color = class_color_mapping.get(label, (255, 255, 255))  # Default to white

        # Draw rectangle on the image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

    # Convert the image to RGB for displaying with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")

    # Save the figure
    output_path = os.path.join(output_dir, filename)
    save_high_res_plot(output_path)


def classify_image(image_path):

    base_output_dir = (
        "/home/etaylor/code_projects/thesis/src/pipelines/end_to_end/results"
    )
    output_dir = configure_output_directory(image_path, base_output_dir)

    # define configs for models
    detection_model_config = {
        "model_name": "faster_rcnn_R_50_C4_1x",
        "checkpoint": "/home/etaylor/code_projects/thesis/checkpoints/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/29-04-2024_16-09-41/model_final.pth",
        "yaml_file": "/home/etaylor/code_projects/thesis/checkpoints/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/29-04-2024_16-09-41/config.yaml",
    }

    classification_models_config = {
        "trichome_classification": {
            "checkpoint": "/home/etaylor/code_projects/thesis/checkpoints/trichome_image_classification/yolo/YOLOv11/Nano_dataset_0.pt"
        },
        "blur_classification": {
            "checkpoint": "/home/etaylor/code_projects/thesis/checkpoints/blur_image_classification/yolo/after_tuning/YOLOv8/Nano_dataset_0.pt"
        },
    }

    # define patch_size
    patch_size = 512
    model_type = "yolo"
    classify_blurry_flag = False

    # load detection model
    detection_model = load_obj_detection_model(detection_model_config, patch_size)

    # load blur classification model
    blur_classification_model = load_classification_model(
        model_config=classification_models_config["blur_classification"],
        model_type=model_type,
    )

    trichome_classification_model = load_classification_model(
        model_config=classification_models_config["trichome_classification"],
        model_type=model_type,
    )

    # Start time measurement
    start_time = time.time()

    # run object detection model on the image
    object_detection_results = perform_trichome_detection(image_path, detection_model)

    # End time measurement
    object_detection_time = time.time() - start_time

    # Plot current detections after object detection
    plot_image(
        image_path,
        object_detection_results.object_prediction_list,
        output_dir,
        title="After Object Detection",
        filename="detections_after_object_detection.png",
    )

    # filter large objects
    filtered_predictions = filter_large_objects(
        object_detection_results.object_prediction_list
    )

    # After performing trichome detection, apply NMS to remove redundant detections
    filtered_predictions = non_max_suppression(filtered_predictions, iou_threshold=0.7)
    object_detection_results.object_prediction_list = filtered_predictions

    # Start time measurement
    start_time = time.time()

    # Filter blurry objects
    filtered_predictions, blurry_trichomes = perform_blur_classification(
        image_path,
        object_detection_results.object_prediction_list,
        blur_classification_model,
        model_type,
    )
    object_detection_results.object_prediction_list = filtered_predictions

    # End time measurement
    filter_blurry_objects_time = time.time() - start_time

    if classify_blurry_flag is True:
        # Process and plot blurry trichomes
        classify_blurry_trichomes(
            blurry_trichomes,
            trichome_classification_model,
            output_dir,
            model_type=model_type,
        )

        # Plot blurry detections on the full image
        plot_image(
            image_path,
            blurry_trichomes,
            output_dir,
            title="Blurry Trichomes Filtered Out",
            filename="blurry_trichomes_filtered_out.png",
        )

    # Plot current detections after filtering blurry objects
    plot_image(
        image_path,
        object_detection_results.object_prediction_list,
        output_dir,
        title="After Filtering Blurry Trichomes",
        filename="detections_after_filtering_blurry_trichomes.png",
    )

    # Start time measurement
    start_time = time.time()

    # run classification model on the filtered predictions
    classify_trichomes(
        image_path,
        object_detection_results,
        trichome_classification_model,
        output_dir,
        model_type,
    )

    # End time measurement
    classify_objects_time = time.time() - start_time

    # Plot current detections after classification
    plot_image(
        image_path,
        object_detection_results.object_prediction_list,
        output_dir,
        title="With Trichome Classification",
        filename="detections_with_classification.png",
    )

    # calc distribution of the trichome
    trichome_distributions = calc_trichome_distribution(
        object_detection_results.object_prediction_list
    )

    # Log the time measurements
    print("Object Detection Time:", object_detection_time)
    print("Filter Blurry Objects Time:", filter_blurry_objects_time)
    print("Classify Objects Time:", classify_objects_time)
    print(
        f"Total Time: {object_detection_time + filter_blurry_objects_time + classify_objects_time}"
    )

    print("Trichome distribution:\n", trichome_distributions)


if __name__ == "__main__":
    image_path = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_1/images/day_9_2024_06_27/greenhouse/268/IMG_8593.JPG"
    classify_image(image_path)
