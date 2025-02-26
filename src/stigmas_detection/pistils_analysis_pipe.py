import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# =============================================================================
# 1. Model Loading Functions
# =============================================================================


def load_yolo_model(checkpoint_path):
    """Loads a YOLO model from the given checkpoint."""
    return YOLO(checkpoint_path)


# =============================================================================
# 2. Pistil Segmentation Functions
# =============================================================================


def segment_pistils(image, model):
    """
    Performs inference on the image using the pistil segmentation model.

    Returns:
        result: The first prediction result from the model.
    """
    results = model.predict(image)
    return results[0]


def extract_segmented_objects(image_rgb, result, save_dir=None):
    """
    Extracts individual segmented objects from the model result.

    Args:
        image_rgb: The original image in RGB.
        result: The prediction result from the YOLO model.
        save_dir: Optional directory to save each segmented object.

    Returns:
        extracted_segments: List of cropped segmented objects.
    """
    masks = result.masks.data.cpu().numpy()  # Extract masks
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes for cropping

    if len(masks) == 0:
        print("No objects detected.")
        return []

    extracted_segments = []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i, mask in enumerate(masks):
        # Resize mask to image size and binarize
        resized_mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))
        binary_mask = (resized_mask > 0.5).astype(np.uint8)

        # Apply mask to the image
        segmented_object = cv2.bitwise_and(image_rgb, image_rgb, mask=binary_mask)

        # Crop using the bounding box
        x_min, y_min, x_max, y_max = map(int, boxes[i])
        cropped_object = segmented_object[y_min:y_max, x_min:x_max]
        extracted_segments.append(cropped_object)

        if save_dir:
            save_path = os.path.join(save_dir, f"segment_{i + 1}.png")
            cv2.imwrite(save_path, cv2.cvtColor(cropped_object, cv2.COLOR_RGB2BGR))
            print(f"Saved segmented object: {save_path}")

    return extracted_segments


def visualize_segmented_objects(segmented_objects):
    """Displays segmented objects side by side."""
    if not segmented_objects:
        print("No segmented objects to display.")
        return

    fig, axes = plt.subplots(1, len(segmented_objects), figsize=(16, 8))
    for idx, obj in enumerate(segmented_objects):
        if len(segmented_objects) > 1:
            axes[idx].imshow(obj)
            axes[idx].set_title(f"Segment {idx + 1}")
            axes[idx].axis("off")
        else:
            axes.imshow(obj)
            axes.set_title("Segment 1")
            axes.axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
# 3. Color Semantic Segmentation Functions (using YOLO)
# =============================================================================


def run_color_semantic_segmentation(image, model):
    """
    Runs the color semantic segmentation model on the image.

    Returns:
        result: The first prediction result from the color segmentation model.
    """
    results = model.predict(image)
    return results[0]


def visualize_color_segmentation(original_rgb, annotated_rgb):
    """Displays the original image and its color segmentation side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(annotated_rgb)
    axes[1].set_title("Color Semantic Segmentation")
    axes[1].axis("off")
    plt.show()


# =============================================================================
# 4. Fallback Color Range Segmentation Functions
# =============================================================================


def classify_stigma_colors(segmented_object):
    """
    Classifies pixels in a segmented object into green, white, or orange,
    ensuring mutually exclusive masks with priority: Orange > White > Green.

    Returns:
        Tuple of ratios: (green_ratio, white_ratio, orange_ratio)
    """
    hsv_image = cv2.cvtColor(segmented_object, cv2.COLOR_RGB2HSV)

    # Define HSV ranges (tweak these as needed)
    lower_green = np.array([25, 30, 30], dtype=np.uint8)
    upper_green = np.array([95, 255, 255], dtype=np.uint8)
    lower_white = np.array([0, 0, 220], dtype=np.uint8)
    upper_white = np.array([180, 30, 255], dtype=np.uint8)
    lower_orange = np.array([0, 50, 50], dtype=np.uint8)
    upper_orange = np.array([40, 255, 255], dtype=np.uint8)

    # Create raw masks
    green_mask_raw = cv2.inRange(hsv_image, lower_green, upper_green)
    white_mask_raw = cv2.inRange(hsv_image, lower_white, upper_white)
    orange_mask_raw = cv2.inRange(hsv_image, lower_orange, upper_orange)

    # Morphological cleanup (optional)
    kernel = np.ones((3, 3), np.uint8)
    green_mask_raw = cv2.morphologyEx(green_mask_raw, cv2.MORPH_CLOSE, kernel)
    white_mask_raw = cv2.morphologyEx(white_mask_raw, cv2.MORPH_CLOSE, kernel)
    orange_mask_raw = cv2.morphologyEx(orange_mask_raw, cv2.MORPH_CLOSE, kernel)

    # Priority 1: Orange (all orange pixels)
    orange_exclusive = orange_mask_raw.copy()

    # Priority 2: White (only where not orange)
    not_orange = cv2.bitwise_not(orange_exclusive)
    white_exclusive = cv2.bitwise_and(white_mask_raw, not_orange)

    # Priority 3: Green (only where not orange or white)
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


def apply_color_filter(segmented_object):
    """
    Applies a color filter to remove the green background from the segmented object.

    Returns:
        stigma_mask: The inverted mask where the green background is removed.
    """
    hsv_image = cv2.cvtColor(segmented_object, cv2.COLOR_RGB2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    stigma_mask = cv2.bitwise_not(green_mask)
    return stigma_mask


def apply_mask_on_image(image, mask):
    """Applies a binary mask on the image."""
    return cv2.bitwise_and(image, image, mask=mask)


def show_images(images, titles, figsize=(20, 10)):
    """Displays a list of images with corresponding titles."""
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def full_pipeline_color_range(segmented_object):
    """
    Full fallback pipeline for color ratio computation using color ranges.

    Returns:
        Tuple of color ratios: (green_ratio, white_ratio, orange_ratio)
    """
    # 1. Remove green background
    stigma_mask = apply_color_filter(segmented_object)
    # 2. Extract stigma region
    stigma_extracted = apply_mask_on_image(segmented_object, stigma_mask)
    # 3. Classify colors
    green_ratio, white_ratio, orange_ratio = classify_stigma_colors(stigma_extracted)

    # Optional: visualize the original and processed images with the ratios.
    color_distribution = (
        f"Green: {green_ratio * 100:.2f}%\n"
        f"White: {white_ratio * 100:.2f}%\n"
        f"Orange: {orange_ratio * 100:.2f}%"
    )
    show_images(
        [segmented_object, stigma_extracted],
        ["Original Segmented Object", f"Stigma (No Green)\n{color_distribution}"],
        figsize=(12, 6),
    )

    return green_ratio, white_ratio, orange_ratio


# =============================================================================
# 5. Main Pipeline
# =============================================================================


def main_pipeline(
    image_path, pistils_model_checkpoint, color_model_checkpoint, save_dir=None
):
    # --- Load Models ---
    pistils_model = load_yolo_model(pistils_model_checkpoint)
    color_model = load_yolo_model(color_model_checkpoint)

    # --- Load and Prepare Image ---
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- Pistil Segmentation ---
    print(f"Performing pistil segmentation on image: {os.path.basename(image_path)}")
    pistils_result = segment_pistils(image, pistils_model)
    annotated_pistils = pistils_result.plot()
    annotated_pistils = cv2.cvtColor(annotated_pistils, cv2.COLOR_BGR2RGB)

    # Display original vs. annotated image
    _, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(annotated_pistils)
    axes[1].set_title("Pistils Segmentation")
    axes[1].axis("off")
    plt.show()

    # --- Extract Segmented Pistils ---
    segmented_objects = extract_segmented_objects(
        image_rgb, pistils_result, save_dir=save_dir
    )
    visualize_segmented_objects(segmented_objects)

    # --- Process Each Pistil ---
    for idx, seg_obj in enumerate(segmented_objects):
        print(f"\nProcessing segmented pistil {idx + 1}")

        # 1. Color Semantic Segmentation using YOLO
        color_result = run_color_semantic_segmentation(seg_obj, color_model)
        annotated_color = color_result.plot()
        annotated_color = cv2.cvtColor(annotated_color, cv2.COLOR_BGR2RGB)

        # Visualize the color segmentation result
        visualize_color_segmentation(seg_obj, annotated_color)

        # 2. Fallback: Color Ratio Calculation using Color Ranges
        green_ratio, white_ratio, orange_ratio = full_pipeline_color_range(seg_obj)
        print(
            f"Segment {idx+1} ratios (Fallback): "
            f"Green: {green_ratio*100:.2f}%, "
            f"White: {white_ratio*100:.2f}%, "
            f"Orange: {orange_ratio*100:.2f}%"
        )


if __name__ == "__main__":
    # Define checkpoint paths and input image
    pistils_model_checkpoint = "/home/etaylor/code_projects/thesis/checkpoints/stigmas_segmentation/yolo/fine_tuned/yolov8s_seg_fine_tuned.pt"
    color_model_checkpoint = "/home/etaylor/code_projects/thesis/checkpoints/stigmas_color_segmentation/yolo/fine_tuned/yolov8n-seg_fine_tuned.pt"
    image_path = "/home/etaylor/code_projects/thesis/segments/etaylor_stigmas_dataset/yolo_formatted/images/val/day_4_IMG_5942.jpg"

    # Optional: directory to save segmented objects
    save_dir = "/home/etaylor/code_projects/thesis/segmented_pistils"

    # Run the full pipeline
    main_pipeline(
        image_path, pistils_model_checkpoint, color_model_checkpoint, save_dir
    )
