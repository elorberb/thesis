import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def extract_segmented_objects(image_rgb, result, save_dir=None):
    """
    Given a YOLO result with segmentation masks, extract each segmented pistil.
    Optionally save each cropped pistil in save_dir.
    Returns a list of cropped images.
    """
    if result.masks is None:
        logger.warning("No segmentation masks found in the result.")
        return []

    masks = result.masks.data.cpu().numpy()  # shape: [N, H, W]
    boxes = result.boxes.xyxy.cpu().numpy()  # shape: [N, 4]

    if len(masks) == 0:
        logger.warning("No segmented objects detected.")
        return []

    extracted_segments = []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i, mask in enumerate(masks):
        resized_mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))
        binary_mask = (resized_mask > 0.5).astype(np.uint8)
        segmented_object = cv2.bitwise_and(image_rgb, image_rgb, mask=binary_mask)
        x_min, y_min, x_max, y_max = map(int, boxes[i])
        cropped_object = segmented_object[y_min:y_max, x_min:x_max]
        extracted_segments.append(cropped_object)
        if save_dir:
            save_path = os.path.join(save_dir, f"segment_{i+1}.png")
            cv2.imwrite(save_path, cv2.cvtColor(cropped_object, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved segmented object: {save_path}")
    return extracted_segments


def classify_stigma_colors(segmented_object):
    """
    Computes pixel ratios for orange, white, and green in the segmented image.
    Returns (green_ratio, white_ratio, orange_ratio) as floats in the range [0, 1].
    """
    hsv_image = cv2.cvtColor(segmented_object, cv2.COLOR_RGB2HSV)
    lower_green = np.array([35, 40, 40], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)
    lower_white = np.array([0, 0, 220], dtype=np.uint8)
    upper_white = np.array([180, 30, 255], dtype=np.uint8)
    lower_orange = np.array([10, 80, 70], dtype=np.uint8)
    upper_orange = np.array([25, 255, 255], dtype=np.uint8)

    green_mask_raw = cv2.inRange(hsv_image, lower_green, upper_green)
    white_mask_raw = cv2.inRange(hsv_image, lower_white, upper_white)
    orange_mask_raw = cv2.inRange(hsv_image, lower_orange, upper_orange)

    kernel = np.ones((3, 3), np.uint8)
    green_mask_raw = cv2.morphologyEx(green_mask_raw, cv2.MORPH_CLOSE, kernel)
    white_mask_raw = cv2.morphologyEx(white_mask_raw, cv2.MORPH_CLOSE, kernel)
    orange_mask_raw = cv2.morphologyEx(orange_mask_raw, cv2.MORPH_CLOSE, kernel)

    # Priority assignment: Orange > White > Green
    orange_exclusive = orange_mask_raw.copy()
    not_orange = cv2.bitwise_not(orange_exclusive)
    white_exclusive = cv2.bitwise_and(white_mask_raw, not_orange)
    orange_or_white = cv2.bitwise_or(orange_exclusive, white_exclusive)
    not_orange_or_white = cv2.bitwise_not(orange_or_white)
    green_exclusive = cv2.bitwise_and(green_mask_raw, not_orange_or_white)

    total_union = cv2.bitwise_or(
        orange_exclusive, cv2.bitwise_or(white_exclusive, green_exclusive)
    )
    total_pixels = cv2.countNonZero(total_union)
    orange_count = cv2.countNonZero(orange_exclusive)
    white_count = cv2.countNonZero(white_exclusive)
    green_count = cv2.countNonZero(green_exclusive)

    if total_pixels > 0:
        orange_ratio = orange_count / float(total_pixels)
        white_ratio = white_count / float(total_pixels)
        green_ratio = green_count / float(total_pixels)
    else:
        orange_ratio = white_ratio = green_ratio = 0

    return green_ratio, white_ratio, orange_ratio


def run_stigma_segmentation_pipeline(image_path, model, save_dir=None):
    """
    Runs the full stigma segmentation pipeline:
    1. Loads the image.
    2. Runs YOLO segmentation to detect pistils.
    3. Optionally saves an annotated image.
    4. Extracts segmented pistils and computes color ratios.
    5. Saves the computed ratios to a JSON file.
    Returns the segmented objects and their ratios.
    """
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Error loading image from {image_path}")
        return [], []

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.predict(image)
    result = results[0]

    # Save annotated image (if desired)
    annotated_image = result.plot()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        axes[1].imshow(annotated_image)
        axes[1].set_title("Annotated Image")
        axes[1].axis("off")
        plt.tight_layout()
        annotated_path = os.path.join(save_dir, "annotated.png")
        plt.savefig(annotated_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved annotated image: {annotated_path}")

    # Extract segmented pistils
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    segmentation_save_dir = (
        os.path.join(save_dir, f"{base_name}_segments") if save_dir else None
    )
    segmented_objects = extract_segmented_objects(image_rgb, result, save_dir=segmentation_save_dir)

    # Compute color ratios for each segment (ratios are in 0.x form)
    ratios_list = []
    for idx, segment in enumerate(segmented_objects):
        green_ratio, white_ratio, orange_ratio = classify_stigma_colors(segment)
        ratios_list.append(
            {
                "segment_index": idx + 1,
                "green_ratio": green_ratio,
                "white_ratio": white_ratio,
                "orange_ratio": orange_ratio,
            }
        )
        logger.info(
            f"Segment {idx+1} Color Ratios: Green: {green_ratio:.2f}, White: {white_ratio:.2f}, Orange: {orange_ratio:.2f}"
        )

    # Save ratios to JSON if save_dir is provided
    if save_dir:
        json_path = os.path.join(save_dir, f"{base_name}_stigma_ratios.json")
        with open(json_path, "w") as fp:
            json.dump(ratios_list, fp, indent=4)
        logger.info(f"Saved stigma ratios JSON: {json_path}")

    return segmented_objects, ratios_list


def process_all_folders_stigmas(parent_folder_path, stigma_model, output_dir):
    """
    Process each subfolder and image for stigma segmentation.
    All stigma outputs (annotated images, segmented pistils, and JSON files) are saved under
    output_dir/stigmas/<subfolder>/<image_name>.
    In addition, each segmented pistil image is saved (with a unique name) in
    output_dir/distinct_stigmas.
    """
    logger.info("Processing stigma segmentation for all folders.")
    stigmas_output_dir = os.path.join(output_dir, "stigmas")
    os.makedirs(stigmas_output_dir, exist_ok=True)
    distinct_stigmas_output_dir = os.path.join(output_dir, "distinct_stigmas")
    os.makedirs(distinct_stigmas_output_dir, exist_ok=True)

    subfolders = [f.path for f in os.scandir(parent_folder_path) if f.is_dir()]
    for folder_path in subfolders:
        folder_name = os.path.basename(folder_path)
        folder_output_dir = os.path.join(stigmas_output_dir, folder_name)
        os.makedirs(folder_output_dir, exist_ok=True)

        image_files = [
            f
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
            and f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            base_file_name = os.path.splitext(image_file)[0]
            image_output_dir = os.path.join(folder_output_dir, base_file_name)
            os.makedirs(image_output_dir, exist_ok=True)

            # Run the stigma segmentation pipeline
            segmented_objects, ratios_list = run_stigma_segmentation_pipeline(
                image_path, stigma_model, save_dir=image_output_dir
            )

            # Save each segmented pistil image in distinct_stigmas
            for idx, seg_img in enumerate(segmented_objects):
                seg_filename = f"{base_file_name}_segment_{idx+1}.png"
                seg_path = os.path.join(distinct_stigmas_output_dir, seg_filename)
                cv2.imwrite(seg_path, cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
                logger.info(f"Saved distinct segmented image: {seg_path}")


# ---------------------------
# Regular Main Entry Point
# ---------------------------
def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Hard-coded parameters (modify these paths as needed)
    parent_input_folder = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_1/images/day_1_2024_05_30/greenhouse"
    output_base_folder = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_1/results/faster_rcnn/day_1_2024_05_30/greenhouse"
    model_checkpoint = "/home/etaylor/code_projects/thesis/checkpoints/stigmas_segmentation/yolo/fine_tuned/yolov8m-seg_fine_tuned.pt"

    # Load the stigma segmentation model
    try:
        stigma_model = YOLO(model_checkpoint)
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        return

    # Process all folders for stigma segmentation
    process_all_folders_stigmas(parent_input_folder, stigma_model, output_base_folder)


if __name__ == "__main__":
    main()
