import os
import cv2
import re
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import argparse

# ---------------- Pistils Data ---------------- #
green_data = [
    (214, 217, 162),
    (202, 207, 149),
    (195, 201, 155),
    (216, 222, 194),
    (195, 199, 140),
    (233, 237, 210),
    (203, 207, 148),
    (192, 196, 120),
    (200, 204, 130),
    (198, 204, 134),
    (208, 213, 159),
    (221, 225, 175),
    (236, 240, 203),
    (212, 218, 180),
    (203, 209, 173),
    (232, 237, 197),
    (198, 202, 151),
    (241, 244, 215),
    (127, 135, 76),
    (130, 134, 73),
    (127, 126, 70),
    (211, 216, 162),
    (216, 219, 166),
    (210, 213, 156),
    (207, 213, 151),
    (220, 224, 173),
    (210, 218, 169),
    (211, 219, 172),
    (205, 214, 167),
    (198, 209, 151),
    (174, 189, 146),
    (206, 212, 174),
    (208, 215, 173),
    (208, 215, 181),
    (208, 214, 168),
    (127, 137, 76),
    (185, 190, 132),
    (154, 143, 87),
    (210, 205, 149),
    (217, 215, 154),
    (218, 221, 168),
    (240, 236, 201),
    (245, 250, 218),
    (218, 223, 159),
    (218, 221, 164),
    (212, 218, 146),
    (212, 219, 139),
    (216, 223, 146),
    (206, 209, 138),
    (195, 193, 118),
    (211, 209, 158),
    (223, 221, 173),
    (212, 211, 167),
]

orange_data = [
    (136, 90, 77),
    (95, 47, 27),
    (135, 99, 67),
    (145, 99, 65),
    (128, 88, 62),
    (141, 100, 56),
    (172, 146, 89),
    (227, 206, 163),
    (225, 198, 145),
    (231, 207, 159),
    (240, 220, 193),
    (214, 193, 148),
    (156, 109, 53),
    (163, 121, 63),
    (150, 99, 46),
    (164, 132, 93),
    (179, 142, 90),
    (150, 105, 50),
    (161, 118, 67),
    (193, 169, 125),
    (206, 192, 165),
    (191, 165, 106),
    (189, 145, 98),
    (213, 169, 124),
    (209, 177, 136),
    (211, 176, 134),
    (209, 179, 143),
    (207, 181, 148),
    (200, 176, 142),
    (192, 163, 131),
    (160, 115, 58),
    (167, 133, 96),
    (167, 127, 68),
    (185, 146, 105),
    (193, 172, 141),
    (141, 99, 74),
    (183, 148, 116),
    (173, 136, 107),
    (185, 154, 123),
    (187, 165, 141),
]


# ---------------- Pistils Pipeline Functions ---------------- #
def load_yolo_model(checkpoint_path):
    """Loads a YOLO model from the given checkpoint path."""
    return YOLO(checkpoint_path)


def segment_pistils(image_bgr, model):
    """
    Runs the pistils segmentation model on a BGR image.
    Returns the first YOLO result object.
    """
    results = model.predict(source=image_bgr, conf=0.3, iou=0.45)
    return results[0]


def extract_segmented_objects(image_rgb, result, save_dir=None):
    """
    Extracts individual pistils from the YOLO segmentation result.
    Optionally saves each segmented pistil to disk.
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
        # Resize mask to match original image size and binarize
        resized_mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))
        binary_mask = (resized_mask > 0.5).astype(np.uint8)
        segmented_object = cv2.bitwise_and(image_rgb, image_rgb, mask=binary_mask)
        x_min, y_min, x_max, y_max = map(int, boxes[i])
        cropped_object = segmented_object[y_min:y_max, x_min:x_max]
        extracted_segments.append(cropped_object)
        if save_dir:
            save_path = os.path.join(save_dir, f"pistil_{i+1}.png")
            cv2.imwrite(save_path, cv2.cvtColor(cropped_object, cv2.COLOR_RGB2BGR))
            print(f"Saved extracted pistil: {save_path}")
    return extracted_segments


def classify_pistil_colors_with_knn(
    pistil_bgr,
    green_data,
    orange_data,
    black_thresh=10,
    white_thresh=240,
    save_images=False,
    output_dir=None,
    save_filename=None,
):
    """
    Classifies each pixel as green or orange using 1-NN in RGB space,
    excluding near-black or near-white background pixels.
    Optionally saves a visualization (the pistil and its classified mask)
    to the given output directory with the provided filename.
    """
    pistil_rgb = cv2.cvtColor(pistil_bgr, cv2.COLOR_BGR2RGB)
    h, w = pistil_rgb.shape[:2]
    bg_mask_black = np.all(pistil_rgb < black_thresh, axis=-1)
    bg_mask_white = np.all(pistil_rgb > white_thresh, axis=-1)
    bg_mask = bg_mask_black | bg_mask_white

    pistil_flat = pistil_rgb.reshape(-1, 3).astype(np.float32)
    bg_mask_flat = bg_mask.reshape(-1)
    valid_indices = np.where(bg_mask_flat == False)[0]
    if len(valid_indices) == 0:
        print("No valid (non-BG) pixels found!")
        return 0.0, 0.0

    valid_pixels = pistil_flat[valid_indices]
    green_arr = np.array(green_data, dtype=np.float32)
    orange_arr = np.array(orange_data, dtype=np.float32)
    pixel_expanded = valid_pixels[:, np.newaxis, :]
    green_expanded = green_arr[np.newaxis, :, :]
    orange_expanded = orange_arr[np.newaxis, :, :]
    dist2_green = np.sum((pixel_expanded - green_expanded) ** 2, axis=2)
    dist2_orange = np.sum((pixel_expanded - orange_expanded) ** 2, axis=2)
    min_dist2_green = np.min(dist2_green, axis=1)
    min_dist2_orange = np.min(dist2_orange, axis=1)
    is_green_valid = min_dist2_green < min_dist2_orange
    is_orange_valid = ~is_green_valid

    labels = np.zeros((h * w,), dtype=np.uint8)
    labels[valid_indices[is_green_valid]] = 1  # green
    labels[valid_indices[is_orange_valid]] = 2  # orange
    labels_2d = labels.reshape(h, w)

    green_count = np.count_nonzero(labels_2d == 1)
    orange_count = np.count_nonzero(labels_2d == 2)
    total_valid = green_count + orange_count
    green_ratio = green_count / total_valid
    orange_ratio = orange_count / total_valid

    # Build a visualization image (color-coded)
    classified_img = np.zeros_like(pistil_rgb, dtype=np.uint8)
    classified_img[labels_2d == 1] = (0, 255, 0)  # green
    classified_img[labels_2d == 2] = (255, 165, 0)  # orange

    if save_images and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(pistil_rgb)
        axes[0].set_title("Original Pistil (RGB)")
        axes[0].axis("off")
        axes[1].imshow(classified_img)
        axes[1].set_title("Classified (1-NN)")
        axes[1].axis("off")
        plt.tight_layout()
        # Use provided filename or default name if not given
        vis_filename = save_filename if save_filename else "classified_vis.png"
        vis_path = os.path.join(output_dir, vis_filename)
        plt.savefig(vis_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved classified visualization at: {vis_path}")
    else:
        plt.close("all")

    print(f"Green Ratio (non-BG):  {green_ratio*100:.2f}%")
    print(f"Orange Ratio (non-BG): {orange_ratio*100:.2f}%")
    return green_ratio, orange_ratio


def process_pistils_image(
    image_path, pistils_model, green_data, orange_data, output_dir, save_images=False
):
    """
    Processes one image by:
      - Running pistils segmentation,
      - (Optionally) Saving the annotated full image,
      - Extracting segmented pistils,
      - Running 1-NN color classification for each pistil,
      - Saving all analysis outputs inside a folder named 'pistils_analysis' which contains:
           * extracted_pistils/        (raw segmented pistils as pistil_1.png, etc.)
           * pistils_colors_images/    (color visualization images as pistil_1.png, etc.)
           * annotated_segmentation.png
           * pistils_color_ratios.json (per-pistil results)
           * aggregated_pistils_color_ratios.json (aggregated over all pistils in the image)
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing image: {image_path}")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: cannot load image at {image_path}")
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = segment_pistils(image_bgr, pistils_model)

    # Create a unified analysis directory for pistils analysis inside output_dir
    analysis_dir = os.path.join(output_dir, "pistils_analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # Directories for raw extractions and color images
    extracted_dir = os.path.join(analysis_dir, "extracted_pistils")
    colors_dir = os.path.join(analysis_dir, "pistils_colors_images")
    if save_images:
        os.makedirs(extracted_dir, exist_ok=True)
        os.makedirs(colors_dir, exist_ok=True)

    # Save full annotated segmentation image if requested
    if save_images:
        annotated_bgr = result.plot(labels=False, conf=False)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        annotated_path = os.path.join(analysis_dir, "annotated_segmentation.png")
        cv2.imwrite(annotated_path, cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
        print(f"Annotated segmentation saved at: {annotated_path}")

    # Extract segmented pistils; save in extracted_dir if saving images
    extraction_save_dir = extracted_dir if save_images else None
    pistils = extract_segmented_objects(image_rgb, result, save_dir=extraction_save_dir)

    classification_results = []
    for idx, pistil_crop in enumerate(pistils):
        pistil_crop_bgr = cv2.cvtColor(pistil_crop, cv2.COLOR_RGB2BGR)
        # Save classification visualization in colors_dir with a unique filename
        save_filename = f"pistil_{idx+1}.png" if save_images else None
        g_ratio, o_ratio = classify_pistil_colors_with_knn(
            pistil_crop_bgr,
            green_data,
            orange_data,
            black_thresh=10,
            white_thresh=240,
            save_images=save_images,
            output_dir=colors_dir if save_images else None,
            save_filename=save_filename,
        )
        classification_results.append(
            {"pistil_index": idx + 1, "green_ratio": g_ratio, "orange_ratio": o_ratio}
        )

    # Save the per-pistil classification results JSON in analysis_dir as 'pistils_color_ratios.json'
    results_path = os.path.join(analysis_dir, "pistils_color_ratios.json")
    with open(results_path, "w") as f:
        json.dump(classification_results, f, indent=4)
    print(f"Per-pistil classification results saved at: {results_path}")

    # Aggregate the results (using a simple average here) and save as a separate JSON file
    if classification_results:
        num_pistils = len(classification_results)
        aggregated_green = (
            sum([res["green_ratio"] for res in classification_results]) / num_pistils
        )
        aggregated_orange = (
            sum([res["orange_ratio"] for res in classification_results]) / num_pistils
        )
        aggregated_results = {
            "num_pistils": num_pistils,
            "average_green_ratio": aggregated_green,
            "average_orange_ratio": aggregated_orange,
        }
        aggregated_path = os.path.join(
            analysis_dir, "aggregated_pistils_color_ratios.json"
        )
        with open(aggregated_path, "w") as f:
            json.dump(aggregated_results, f, indent=4)
        print(f"Aggregated classification results saved at: {aggregated_path}")


def process_pistils_images_in_folder(
    folder_path,
    pistils_model,
    green_data,
    orange_data,
    output_base_folder,
    save_images=False,
):
    """
    Processes all images in a folder.
    Creates an output subfolder for each image.
    Skips the first image (flower ID image).
    """
    folder_name = os.path.basename(folder_path)
    folder_output_dir = os.path.join(output_base_folder, folder_name)
    os.makedirs(folder_output_dir, exist_ok=True)
    image_files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    def extract_number(filename):
        match = re.search(r"\d+", filename)
        return int(match.group()) if match else -1

    image_files.sort(key=extract_number)
    # Skip the first image
    for image_file in image_files[1:]:
        image_path = os.path.join(folder_path, image_file)
        base_name = os.path.splitext(image_file)[0]
        image_output_dir = os.path.join(folder_output_dir, base_name)
        os.makedirs(image_output_dir, exist_ok=True)
        process_pistils_image(
            image_path,
            pistils_model,
            green_data,
            orange_data,
            image_output_dir,
            save_images=save_images,
        )


def process_all_folders(
    parent_folder,
    pistils_model,
    green_data,
    orange_data,
    output_base_folder,
    save_images=False,
):
    """
    Processes all subfolders in the parent folder.
    """
    subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]
    start_time = time.time()
    for folder in subfolders:
        print(f"Processing folder: {folder}")
        process_pistils_images_in_folder(
            folder,
            pistils_model,
            green_data,
            orange_data,
            output_base_folder,
            save_images=save_images,
        )
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")


# ---------------- Main ---------------- #
if __name__ == "__main__":
    # For CLI use, uncomment the following block:
    parser = argparse.ArgumentParser(description="Batch Pistils Pipeline")
    parser.add_argument(
        "--parent_input_folder",
        type=str,
        required=True,
        help="Parent folder path containing subfolders of images",
    )
    parser.add_argument(
        "--output_base_folder",
        type=str,
        required=True,
        help="Base folder to save all results",
    )

    args = parser.parse_args()

    pistils_model_checkpoint = "/home/etaylor/code_projects/thesis/checkpoints/stigmas_segmentation/yolo/fine_tuned/yolov8s_seg_fine_tuned.pt"
    pistils_model = load_yolo_model(pistils_model_checkpoint)
    save_images = True
    process_all_folders(
        args.parent_input_folder,
        pistils_model,
        green_data,
        orange_data,
        args.output_base_folder,
        save_images=save_images,
    )

    # Script version:
    # pistils_model_checkpoint = "/home/etaylor/code_projects/thesis/checkpoints/stigmas_segmentation/yolo/fine_tuned/yolov8s_seg_fine_tuned.pt"
    # save_images = True
    # parent_input_folder = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_1/images/day_4_2024_06_10/greenhouse"
    # output_base_folder = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_1/results/faster_rcnn/day_4_2024_06_10/greenhouse"

    # pistils_model = load_yolo_model(pistils_model_checkpoint)

    # process_all_folders(
    #     parent_input_folder,
    #     pistils_model,
    #     green_data,
    #     orange_data,
    #     output_base_folder,
    #     save_images=save_images,
    # )
