import os
import cv2
import logging
import numpy as np
from fastai.vision.all import *
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO

from PIL import Image as PilImage

from src.pipelines.end_to_end.end_to_end_utils import (
    load_obj_detection_model,
    perform_trichome_detection,
    filter_large_objects,
    extend_bounding_box,
    crop_image,
    non_max_suppression,
    save_visuals,
    save_results,
    compute_class_distribution,
    compute_normalized_class_distribution,
    save_class_distribution,
    compute_aggregated_class_distribution,
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


# ------- Utility Functions -------


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


def classify_trichomes(image_path, result, classification_model, model_type):
    logger.info("Classifying good quality detected trichomes.")
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    # Mapping from classification model class IDs to detection model class IDs
    classification_to_detection_id_mapping = {
        0: 3,
        1: 1,
        2: 2,
    }

    # Mapping from class IDs to class names for both models
    detection_class_id_to_name = {1: "Clear", 2: "Cloudy", 3: "Amber"}
    classification_class_id_to_name = {0: "amber", 1: "clear", 2: "cloudy"}

    for _, prediction in enumerate(result.object_prediction_list):

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
        # if isinstance(cropped_image, np.ndarray):
        #     cropped_pil_image = PILImage.create(cropped_image)
        # else:
        #     cropped_pil_image = cropped_image

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


def save_image(image_path, predictions, output_dir, title, filename):
    # Read the image
    image = cv2.imread(image_path)

    image_name = os.path.basename(image_path).split("/")[-1]

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
    plt.title(title + f" for image {image_name}")
    plt.axis("off")

    # Save the figure
    output_path = os.path.join(output_dir, filename)
    save_high_res_plot(output_path)


def process_image(
    image_path,
    detection_model,
    classification_models,
    model_type,
    patch_size,
    perform_blur_classification_flag=False,
):

    # run object detection model on the image
    results = perform_trichome_detection(image_path, detection_model, patch_size)

    print(results.object_prediction_list[0])
    # filter large objects
    filtered_predictions = filter_large_objects(results.object_prediction_list)

    # After performing trichome detection, apply NMS to remove redundant detections
    filtered_predictions = non_max_suppression(filtered_predictions, iou_threshold=0.7)
    results.object_prediction_list = filtered_predictions

    if perform_blur_classification_flag is True:

        # Filter blurry objects
        filtered_predictions, _ = perform_blur_classification(
            image_path,
            results.object_prediction_list,
            classification_models["blur_classification"],
            model_type,
        )
        results.object_prediction_list = filtered_predictions

    # run classification model on predictions and modify classes in memory
    classify_trichomes(
        image_path,
        results,
        classification_models["trichome_classification"],
        model_type,
    )

    return results


def process_images_in_folder(
    folder_path,
    detection_model,
    classification_models,
    output_dir,
    patch_size,
    model_type,
    blur_classification_flag,
):
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

        results = process_image(
            image_path,
            detection_model,
            classification_models,
            model_type,
            patch_size,
            perform_blur_classification_flag=blur_classification_flag,
        )

        # Export results with filtered predictions
        save_visuals(results, image_output_dir, base_file_name)
        save_results(results, image_output_dir, base_file_name)

        # Compute class label distribution and normalized distribution for the image
        class_distribution = compute_class_distribution(results.object_prediction_list)
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

        aggregated_results.extend(results.object_prediction_list)

        # Calculate total processing time
    total_time = time.time() - start_time
    logger.info(f"Total time taken to process the folder: {total_time:.2f} seconds")

    # Compute and save aggregated class label distribution for all images
    compute_aggregated_class_distribution(aggregated_results, result_dir)


if __name__ == "__main__":

    # define configs for models
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

    # define patch_size
    patch_size = 512
    model_type = "yolo"
    perform_blur_classification_flag = True

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

    classification_models = {
        "trichome_classification": trichome_classification_model,
        "blur_classification": blur_classification_model,
    }

    image_path = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_1/images/day_7_2024_06_20/greenhouse/202/IMG_7745.JPG"
    folder_path = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_1/images/day_7_2024_06_20/greenhouse/186"

    folder_name = os.path.basename(folder_path)
    output_dir = "/home/etaylor/code_projects/thesis/src/pipelines/end_to_end/results"

    # process folder
    process_images_in_folder(
        folder_path=folder_path,
        detection_model=detection_model,
        classification_models=classification_models,
        output_dir=output_dir,
        patch_size=patch_size,
        model_type=model_type,
        blur_classification_flag=perform_blur_classification_flag,
    )

    # results = process_image(
    #     image_path=image_path,
    #     detection_model=detection_model,
    #     classification_models=classification_models,
    #     model_type=model_type,
    #     patch_size=patch_size,
    #     perform_blur_classification_flag=perform_blur_classification_flag,
    # )

    # # Compute class label distribution and normalized distribution for the image
    # class_distribution = compute_class_distribution(results.object_prediction_list)
    # normalized_class_distribution = compute_normalized_class_distribution(
    #     class_distribution
    # )

    # # Save combined distributions to JSON
    # distribution_json_path = os.path.join(output_dir, "class_distribution.json")
    # save_class_distribution(
    #     class_distribution, normalized_class_distribution, distribution_json_path
    # )

    # save_visuals(results, output_dir, "test")
