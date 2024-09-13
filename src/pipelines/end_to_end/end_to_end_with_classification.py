import os
from end_to_end_pipe import load_obj_detection_model
import cv2
import logging
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np
from fastai.vision.all import *
import matplotlib.pyplot as plt
import time

# Remove conflicting import
# from PIL import Image as PILImage
# If needed, import PIL.Image with a different alias
from PIL import Image as PilImage

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


def load_classification_model(classification_model_config):
    print("Loading the classification model.")
    learn = load_learner(classification_model_config["checkpoint"])
    return learn


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


def classify_cropped_image(cropped_image, classification_model):
    # Convert the image to a format acceptable by the model
    if isinstance(cropped_image, np.ndarray):
        cropped_pil_image = PILImage.create(cropped_image)
    else:
        cropped_pil_image = cropped_image

    pred_class, pred_idx, probs = classification_model.predict(cropped_pil_image)
    return pred_class


def plot_classified_object(
    cropped_image, detection_class_name, classification_class_name, output_dir, index, prefix="classified_object"
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
    plt.savefig(output_path)
    plt.close()


def classify_objects(image_path, result, classification_model, output_dir):
    logger.info("Classifying detected objects.")
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
        print(f"Detection model pred id before: {detection_model_pred_class_id}")

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
        classification_model_pred_class_name = classify_cropped_image(
            cropped_pil_image, classification_model
        )

        # Convert class name to class ID in the classification model
        classification_model_pred_class_name_lower = (
            classification_model_pred_class_name.lower()
        )
        classification_model_pred_class_id = list(
            classification_class_id_to_name.keys()
        )[
            list(classification_class_id_to_name.values()).index(
                classification_model_pred_class_name_lower
            )
        ]

        # Map classification model class ID to detection model class ID
        detection_model_class_id = classification_to_detection_id_mapping.get(
            classification_model_pred_class_id, 0
        )

        # Update the prediction category id and name
        prediction.category.id = detection_model_class_id
        prediction.category.name = detection_class_id_to_name.get(
            detection_model_class_id, "Unknown"
        )
        print(f"Detection model pred id after mapping: {prediction.category.id}")

        # Get class names for logging and plotting
        faster_rcnn_class_name = detection_class_id_to_name.get(
            detection_model_pred_class_id, "Unknown"
        )
        classification_class_name = classification_class_id_to_name.get(
            classification_model_pred_class_id, "Unknown"
        )

        # Plot the classified object
        plot_classified_object(
            cropped_pil_image,
            faster_rcnn_class_name,
            classification_class_name,
            output_dir,
            idx,
        )

        logger.info(
            f"Detected Class (Faster R-CNN): {detection_model_pred_class_id} {faster_rcnn_class_name}"
        )
        logger.info(
            f"Predicted Class (Classification Model): {classification_model_pred_class_id} {classification_class_name}"
        )

    end_classification = time.time() - start_classification
    logger.info(f"Time taken for classification: {end_classification:.2f} seconds")


def filter_blurry_objects(image_path, predictions, blur_classification_model):
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

        # Convert to fastai's PILImage
        if isinstance(cropped_image, np.ndarray):
            cropped_pil_image = PILImage.create(cropped_image)
        else:
            cropped_pil_image = cropped_image

        # Classify for blur detection
        is_sharp, _, _ = blur_classification_model.predict(cropped_pil_image)

        if is_sharp == "good_quality":
            filtered_predictions.append(prediction)
        else:
            blurry_trichomes.append((prediction, cropped_pil_image))

    logger.info(
        f"Number of predictions after blur filtering: {len(filtered_predictions)}"
    )
    logger.info(
        f"Number of filtered predictions: {len(predictions) - len(filtered_predictions)}"
    )
    return filtered_predictions, blurry_trichomes


def classify_and_plot_blurry_trichomes(blurry_trichomes, classification_model, output_dir):
    logger.info("Classifying and plotting blurry trichomes.")
    # Mapping from classification model class IDs to class names
    classification_class_id_to_name = {0: "amber", 1: "clear", 2: "cloudy"}

    for idx, (prediction, cropped_pil_image) in enumerate(blurry_trichomes):
        # Classify the cropped image using the classification model
        classification_model_pred_class_name = classify_cropped_image(
            cropped_pil_image, classification_model
        )

        # Convert class name to class ID in the classification model
        classification_model_pred_class_name_lower = (
            classification_model_pred_class_name.lower()
        )
        classification_model_pred_class_id = list(
            classification_class_id_to_name.keys()
        )[
            list(classification_class_id_to_name.values()).index(
                classification_model_pred_class_name_lower
            )
        ]

        # Get class names for logging and plotting
        classification_class_name = classification_class_id_to_name.get(
            classification_model_pred_class_id, "Unknown"
        )

        # Plot the classified object
        plot_classified_object(
            cropped_pil_image,
            "Blurry",
            classification_class_name,
            output_dir,
            idx,
            prefix="blurry_trichome"
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

    # Correct class IDs to class names mapping
    labels = {1: "Clear", 2: "Cloudy", 3: "Amber"}
    distribution_named = {labels.get(k, "Unknown"): v for k, v in distribution.items()}

    logger.info(f"Trichome Distribution: {distribution_named}")
    return distribution_named


def plot_current_detections(
    image_path,
    predictions,
    output_dir,
    title="Current Detections",
    filename="current_detections.png",
):
    # Read the image
    image = cv2.imread(image_path)
    
    # Define color mapping for classes (in BGR format)
    class_color_mapping = {
        'Clear': (0, 0, 255),        # Red
        'Cloudy': (0, 165, 255),     # Orange
        'Amber': (0, 255, 0),        # Green
        'Unknown': (255, 255, 255)   # White
    }
    
    # Initialize counters for each class to number the bounding boxes
    class_counters = {}
    
    # Iterate over predictions
    for prediction in predictions:
        x_min = int(prediction.bbox.minx)
        y_min = int(prediction.bbox.miny)
        x_max = int(prediction.bbox.maxx)
        y_max = int(prediction.bbox.maxy)
        
        # Get the class name or ID
        label = None
        if prediction.category.name:
            label = prediction.category.name.capitalize()
        elif prediction.category.id is not None:
            # Map the class ID to class name if possible
            detection_class_id_to_name = {1: "Clear", 2: "Cloudy", 3: "Amber"}
            label = detection_class_id_to_name.get(prediction.category.id, "Unknown")
        else:
            label = "Unknown"
        
        # Get color for the class
        color = class_color_mapping.get(label, (255, 255, 255))  # Default to white
        
        # Get and increment the counter for the class
        counter = class_counters.get(label, 0) + 1
        class_counters[label] = counter
        
        # Draw rectangle on the image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Put the class label and counter on the bounding box
        cv2.putText(
            image,
            f"{label} {counter}",
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )
    
    # Convert the image to RGB for displaying with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")

    # Save the figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()


def plot_blurry_detections(
    image_path,
    blurry_trichomes,
    output_dir,
    title="Blurry Detections",
    filename="blurry_detections.png",
):
    # Read the image
    image = cv2.imread(image_path)

    # Color for blurry detections (e.g., blue)
    color = (255, 0, 0)  # Blue in BGR format

    # Iterate over blurry trichomes
    for idx, (prediction, _) in enumerate(blurry_trichomes):
        x_min = int(prediction.bbox.minx)
        y_min = int(prediction.bbox.miny)
        x_max = int(prediction.bbox.maxx)
        y_max = int(prediction.bbox.maxy)

        # Draw rectangle on the image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # Put the index as label on the bounding box
        cv2.putText(
            image,
            str(idx + 1),
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )

    # Convert the image to RGB for displaying with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")

    # Save the figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()


def classify_image(image_path):

    # Define the output directory for saving plots
    output_dir = "/home/etaylor/code_projects/thesis/src/pipelines/end_to_end/results"

    # Get the name of the file (without file type)
    file_name = os.path.splitext(os.path.basename(image_path))[0]

    # Append the file name to the output directory
    output_dir = os.path.join(output_dir, file_name)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # define configs for models
    detection_model_config = {
        "model_name": "faster_rcnn_R_50_C4_1x",
        "checkpoint": "/home/etaylor/code_projects/thesis/checkpoints/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/29-04-2024_16-09-41/model_final.pth",
        "yaml_file": "/home/etaylor/code_projects/thesis/checkpoints/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/29-04-2024_16-09-41/config.yaml",
    }

    classification_models_config = {
        "trichome_classification": {
            "checkpoint": "/home/etaylor/code_projects/thesis/checkpoints/trichome_image_classification/alexnet.pkl"
        },
        "blur_classification": {
            "checkpoint": "/home/etaylor/code_projects/thesis/checkpoints/blur_image_classification/alexnet.pkl"
        },
    }

    # define patch_size
    patch_size = 512

    # Start time measurement
    start_time = time.time()

    # load detection model
    detection_model = load_obj_detection_model(detection_model_config, patch_size)

    # End time measurement
    detection_model_time = time.time() - start_time

    # Start time measurement
    start_time = time.time()

    # load blur classification model
    blur_classification_model = load_classification_model(
        classification_models_config["blur_classification"]
    )

    trichome_classification_model = load_classification_model(
        classification_models_config["trichome_classification"]
    )

    # End time measurement
    classification_model_time = time.time() - start_time

    # Start time measurement
    start_time = time.time()

    # run object detection model on the image
    object_detection_results = perform_object_detection(image_path, detection_model)

    # End time measurement
    object_detection_time = time.time() - start_time

    # Plot current detections after object detection
    plot_current_detections(
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
    object_detection_results.object_prediction_list = filtered_predictions

    # Plot current detections after filtering large objects
    plot_current_detections(
        image_path,
        object_detection_results.object_prediction_list,
        output_dir,
        title="After Filtering Large Objects",
        filename="detections_after_filtering_large_objects.png",
    )

    # Start time measurement
    start_time = time.time()

    # Filter blurry objects
    filtered_predictions, blurry_trichomes = filter_blurry_objects(
        image_path,
        object_detection_results.object_prediction_list,
        blur_classification_model,
    )
    object_detection_results.object_prediction_list = filtered_predictions

    # Process and plot blurry trichomes
    classify_and_plot_blurry_trichomes(
        blurry_trichomes, trichome_classification_model, output_dir
    )

    # Plot blurry detections on the full image
    plot_blurry_detections(
        image_path,
        blurry_trichomes,
        output_dir,
        title="Blurry Trichomes Filtered Out",
        filename="blurry_trichomes_filtered_out.png",
    )

    # End time measurement
    filter_blurry_objects_time = time.time() - start_time

    # Plot current detections after filtering blurry objects
    plot_current_detections(
        image_path,
        object_detection_results.object_prediction_list,
        output_dir,
        title="After Filtering Blurry Objects",
        filename="detections_after_filtering_blurry_objects.png",
    )

    # Start time measurement
    start_time = time.time()

    # run classification model on the filtered predictions
    classify_objects(
        image_path, object_detection_results, trichome_classification_model, output_dir
    )

    # End time measurement
    classify_objects_time = time.time() - start_time

    # Plot current detections after classification
    plot_current_detections(
        image_path,
        object_detection_results.object_prediction_list,
        output_dir,
        title="After Classification",
        filename="detections_after_classification.png",
    )

    # Start time measurement
    start_time = time.time()

    # calc distribution of the trichome
    trichome_distribution = calc_trichome_distribution(
        object_detection_results.object_prediction_list
    )

    # End time measurement
    calc_trichome_distribution_time = time.time() - start_time

    # Log the time measurements
    print("Load Object Detection Model Time:", detection_model_time)
    print("Load Classification Models Time:", classification_model_time)
    print("Object Detection Time:", object_detection_time)
    print("Filter Blurry Objects Time:", filter_blurry_objects_time)
    print("Classify Objects Time:", classify_objects_time)
    print("Calc Trichome Distribution Time:", calc_trichome_distribution_time)
    print(
        f"Total Time: {detection_model_time + classification_model_time + object_detection_time + filter_blurry_objects_time + classify_objects_time + calc_trichome_distribution_time}"
    )

    print("Trichome distribution:", trichome_distribution)


if __name__ == "__main__":
    image_path = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/images/day_9_2024_06_27/greenhouse/264/IMG_8745.JPG"
    classify_image(image_path)
