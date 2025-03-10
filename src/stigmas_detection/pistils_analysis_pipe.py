import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# =============================================================================
# 1. YOLO Model Loading
# =============================================================================


def load_yolo_model(checkpoint_path):
    """Loads a YOLO model from the given checkpoint."""
    return YOLO(checkpoint_path)


# =============================================================================
# 2. Pistil Segmentation (YOLO Model #1)
# =============================================================================


def segment_pistils(image_bgr, model):
    """
    Runs the pistil segmentation model on a BGR image.
    Returns the first YOLO result object.
    """
    results = model.predict(image_bgr)
    return results[0]


def extract_segmented_objects(image_rgb, result, save_dir=None):
    """
    Extracts individual pistils from the YOLO segmentation result.

    Args:
        image_rgb:  The original image in RGB format.
        result:     The YOLO result (must have segmentation masks).
        save_dir:   Optional directory to save each extracted pistil.

    Returns:
        A list of cropped pistils (each in RGB).
    """
    if result.masks is None or result.masks.data is None:
        print("No pistils detected.")
        return []

    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()

    extracted_segments = []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i, mask in enumerate(masks):
        # Resize mask to the image size and binarize
        resized_mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))
        binary_mask = (resized_mask > 0.5).astype(np.uint8)

        # Apply mask to the RGB image
        segmented_object = cv2.bitwise_and(image_rgb, image_rgb, mask=binary_mask)

        # Crop using bounding box
        x_min, y_min, x_max, y_max = map(int, boxes[i])
        cropped_object = segmented_object[y_min:y_max, x_min:x_max]
        extracted_segments.append(cropped_object)

        if save_dir:
            save_path = os.path.join(save_dir, f"pistil_{i + 1}.png")
            cv2.imwrite(save_path, cv2.cvtColor(cropped_object, cv2.COLOR_RGB2BGR))
            print(f"Saved pistil: {save_path}")

    return extracted_segments


def visualize_segmented_objects(segmented_objects):
    """Displays each extracted pistil side by side."""
    if not segmented_objects:
        print("No pistils to display.")
        return

    fig, axes = plt.subplots(1, len(segmented_objects), figsize=(16, 8))
    if len(segmented_objects) == 1:
        axes.imshow(segmented_objects[0])
        axes.set_title("Segment 1")
        axes.axis("off")
    else:
        for idx, obj in enumerate(segmented_objects):
            axes[idx].imshow(obj)
            axes[idx].set_title(f"Segment {idx + 1}")
            axes[idx].axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
# 3. Color Segmentation (YOLO Model #2)
# =============================================================================


def run_color_semantic_segmentation(seg_obj_rgb, model):
    """
    Runs the color segmentation model (YOLO) on a pistil in RGB.
    If the model was trained on BGR, we convert from RGB->BGR first.
    """
    image_bgr = cv2.cvtColor(seg_obj_rgb, cv2.COLOR_RGB2BGR)
    results = model.predict(image_bgr)
    return results[0]


def visualize_color_segmentation(original_rgb, annotated_bgr):
    """
    Displays the original pistil (RGB) vs. YOLO's annotated output (BGR).
    """
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Pistil (RGB)")
    axes[0].axis("off")
    axes[1].imshow(annotated_rgb)
    axes[1].set_title("Color Model Segmentation")
    axes[1].axis("off")
    plt.show()


def extract_color_model_masks(result, image_shape):
    """
    Extracts predicted masks for green (class 0) and orange (class 1)
    from the YOLO color segmentation result.

    NOTE: If your classes differ, update the class indices accordingly.
    """
    if result.masks is None or result.masks.data is None:
        # No predictions
        height, width = image_shape[:2]
        empty = np.zeros((height, width), dtype=np.uint8)
        return empty, empty

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    height, width = image_shape[:2]
    pred_green_mask = np.zeros((height, width), dtype=np.uint8)
    pred_orange_mask = np.zeros((height, width), dtype=np.uint8)

    for i, mask in enumerate(masks):
        resized_mask = cv2.resize(mask, (width, height))
        binary_mask = (resized_mask > 0.5).astype(np.uint8) * 255

        # If your color model has 0 => green, 1 => orange:
        if classes[i] == 0:  # green
            pred_green_mask = cv2.bitwise_or(pred_green_mask, binary_mask)
        elif classes[i] == 1:  # orange
            pred_orange_mask = cv2.bitwise_or(pred_orange_mask, binary_mask)

    return pred_green_mask, pred_orange_mask


# =============================================================================
# 4. HSV Fallback (Orange > Green)
# =============================================================================


def hsv_fallback_orange_green(seg_obj_rgb):
    """
    Classifies the entire pistil as either orange or green, with priority orange>green.
    Returns (fallback_orange, fallback_green) as 0/255 masks.

    Adjust HSV ranges to match your data.
    """
    hsv = cv2.cvtColor(seg_obj_rgb, cv2.COLOR_RGB2HSV)

    # Example ranges
    lower_orange = np.array([0, 50, 50], dtype=np.uint8)
    upper_orange = np.array([40, 255, 255], dtype=np.uint8)

    lower_green = np.array([40, 30, 30], dtype=np.uint8)
    upper_green = np.array([95, 255, 255], dtype=np.uint8)

    orange_raw = cv2.inRange(hsv, lower_orange, upper_orange)
    green_raw = cv2.inRange(hsv, lower_green, upper_green)

    # Priority 1: Orange
    fallback_orange = orange_raw.copy()

    # Priority 2: Green => only where not orange
    not_orange = cv2.bitwise_not(fallback_orange)
    fallback_green = cv2.bitwise_and(green_raw, not_orange)

    return fallback_orange, fallback_green


# =============================================================================
# 5. Combine Model + Fallback => Ensure Sum=1
# =============================================================================


def combine_model_and_fallback(
    seg_obj_rgb, object_mask, pred_orange_mask, pred_green_mask, fallback_func
):
    """
    1) Takes the color model's orange/green predictions (priority).
    2) Fallback HSV classification for missing pixels (orange>green).
    3) Removes overlap (orange overrides green).
    4) Forces leftover => green.
    5) Computes final orange/green ratio => sum=1.

    Returns:
      final_orange_mask, final_green_mask, orange_ratio, green_ratio
    """
    # (A) Identify missing area = object_mask - (model union)
    model_union = cv2.bitwise_or(pred_orange_mask, pred_green_mask)
    model_union_in_obj = cv2.bitwise_and(model_union, object_mask)
    missing_mask = cv2.bitwise_and(object_mask, cv2.bitwise_not(model_union_in_obj))

    # (B) Fallback classification only on missing region
    fb_orange_full, fb_green_full = fallback_func(seg_obj_rgb)
    fb_orange = cv2.bitwise_and(fb_orange_full, missing_mask)
    fb_green = cv2.bitwise_and(fb_green_full, missing_mask)

    # (C) Merge with model predictions
    final_orange = cv2.bitwise_or(pred_orange_mask, fb_orange)
    final_green = cv2.bitwise_or(pred_green_mask, fb_green)

    # (D) Remove overlap => ORANGE PRIORITY
    overlap = cv2.bitwise_and(final_orange, final_green)
    if cv2.countNonZero(overlap) > 0:
        # remove overlap from green
        final_green = cv2.bitwise_and(final_green, cv2.bitwise_not(final_orange))

    # (E) Force leftover => green
    union_now = cv2.bitwise_or(final_orange, final_green)
    union_in_obj = cv2.bitwise_and(union_now, object_mask)
    leftover = cv2.bitwise_and(object_mask, cv2.bitwise_not(union_in_obj))
    final_green = cv2.bitwise_or(final_green, leftover)

    # (F) Compute final ratios
    total_pixels = cv2.countNonZero(object_mask)
    orange_count = cv2.countNonZero(final_orange)
    green_count = cv2.countNonZero(final_green)

    if total_pixels > 0:
        orange_ratio = orange_count / float(total_pixels)
        green_ratio = green_count / float(total_pixels)
    else:
        orange_ratio = 0
        green_ratio = 0

    return final_orange, final_green, orange_ratio, green_ratio


# =============================================================================
# 6. Main Pipeline
# =============================================================================


def main_pipeline(
    image_path, pistils_model_checkpoint, color_model_checkpoint, save_dir=None
):
    # (A) Load Models
    pistils_model = load_yolo_model(pistils_model_checkpoint)
    color_model = load_yolo_model(color_model_checkpoint)

    # (B) Read & Show Original Image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: cannot load image at {image_path}")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # (C) Pistil Segmentation
    print(f"Segmenting pistils in: {os.path.basename(image_path)}")
    pistils_result = segment_pistils(image_bgr, pistils_model)
    annotated_bgr = pistils_result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Show original vs. pistil segmentation
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(annotated_rgb)
    axes[1].set_title("Pistils Segmentation")
    axes[1].axis("off")
    plt.show()

    # (D) Extract Pistils
    pistils = extract_segmented_objects(image_rgb, pistils_result, save_dir=save_dir)
    visualize_segmented_objects(pistils)

    # (E) Process Each Pistil
    for i, seg_obj in enumerate(pistils):
        print(f"\nProcessing Pistil {i+1}")

        # 1. Color Model Prediction
        color_result = run_color_semantic_segmentation(seg_obj, color_model)
        annotated_color_bgr = color_result.plot()
        visualize_color_segmentation(seg_obj, annotated_color_bgr)

        # 2. Extract model masks (green/orange)
        height, width = seg_obj.shape[:2]
        pred_green_mask, pred_orange_mask = extract_color_model_masks(
            color_result, (height, width)
        )

        # 3. object_mask = all non-black pixels of the pistil
        gray_obj = cv2.cvtColor(seg_obj, cv2.COLOR_RGB2GRAY)
        _, object_mask = cv2.threshold(gray_obj, 1, 255, cv2.THRESH_BINARY)

        # 4. Combine with fallback => sum=1
        final_orange, final_green, orange_ratio, green_ratio = (
            combine_model_and_fallback(
                seg_obj_rgb=seg_obj,
                object_mask=object_mask,
                pred_orange_mask=pred_orange_mask,
                pred_green_mask=pred_green_mask,
                fallback_func=hsv_fallback_orange_green,
            )
        )

        # 5. Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(seg_obj)
        axes[0].set_title("Segmented Pistil (RGB)")
        axes[0].axis("off")
        axes[1].imshow(final_green, cmap="gray")
        axes[1].set_title("Final Green Mask")
        axes[1].axis("off")
        axes[2].imshow(final_orange, cmap="gray")
        axes[2].set_title("Final Orange Mask")
        axes[2].axis("off")
        plt.show()

        total_ratio = green_ratio + orange_ratio
        print(
            f"Pistil {i+1} => Green: {green_ratio*100:.2f}%, "
            f"Orange: {orange_ratio*100:.2f}% (Sum={total_ratio*100:.2f}%)"
        )


if __name__ == "__main__":
    # Define checkpoint paths and input image
    pistils_model_checkpoint = "/home/etaylor/code_projects/thesis/checkpoints/stigmas_segmentation/yolo/fine_tuned/yolov8s_seg_fine_tuned.pt"
    color_model_checkpoint = "/home/etaylor/code_projects/thesis/checkpoints/stigmas_color_segmentation/yolo/fine_tuned/yolov8n-seg_fine_tuned.pt"
    image_path = "/home/etaylor/code_projects/thesis/segments/etaylor_stigmas_dataset/yolo_formatted/images/val/day_4_IMG_5942.jpg"

    # Optional: directory to save segmented objects
    save_dir = "/home/etaylor/code_projects/thesis/segmented_pistils"

    # Run the full pipeline
    main_pipeline(image_path, pistils_model_checkpoint, color_model_checkpoint)
