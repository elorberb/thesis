import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# -------------------------------
# 1. Inference and Segmentation
# -------------------------------


def extract_segmented_objects(image_rgb, result, image_name, save_dir=None):
    """
    Extract segmented objects from the inference result.
    Resizes each mask to the image dimensions, binarizes it,
    applies it on the image, and then crops the object using the predicted bounding box.

    Now saves a 4-channel PNG (RGBA) so the background is transparent.
    """
    import cv2
    import numpy as np
    import os

    masks = result.masks.data.cpu().numpy()  # Extract masks from result
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes for cropping

    if len(masks) == 0:
        print("No objects detected.")
        return []

    extracted_segments = []

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i, mask in enumerate(masks):
        # 1) Resize mask to match image dimensions and threshold it
        resized_mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))
        binary_mask = (resized_mask > 0.5).astype(np.uint8)

        # 2) Apply the binary mask to the image
        segmented_object = cv2.bitwise_and(image_rgb, image_rgb, mask=binary_mask)

        # 3) Crop the object using the bounding box
        x_min, y_min, x_max, y_max = map(int, boxes[i])
        cropped_object = segmented_object[y_min:y_max, x_min:x_max]
        cropped_mask = binary_mask[y_min:y_max, x_min:x_max]

        # 4) Convert to RGBA (add alpha channel)
        #    Since 'cropped_object' is in RGB, we use COLOR_RGB2RGBA.
        cropped_rgba = cv2.cvtColor(cropped_object, cv2.COLOR_RGB2BGRA)

        # 5) Set alpha channel based on the mask (1 → 255, 0 → 0)
        cropped_rgba[..., 3] = (cropped_mask * 255).astype(np.uint8)

        # 6) Append to the list
        extracted_segments.append(cropped_rgba)

        # 7) Save each segmented object (RGBA PNG) if a directory is provided
        if save_dir:
            save_path = os.path.join(save_dir, f"{image_name}_pistil_{i + 1}.png")
            cv2.imwrite(save_path, cropped_rgba)  # 4-channel PNG
            print(f"Saved segmented object (RGBA): {save_path}")

    return extracted_segments


def visualize_segmented_objects(segmented_objects):
    """
    Visualizes a list of segmented objects using matplotlib.
    """
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
            axes.set_title(f"Segment {idx + 1}")
            axes.axis("off")
    plt.tight_layout()
    plt.show()


def run_inference(image_path, model_checkpoint, save_dir):
    """
    Loads the YOLO model from the given checkpoint, runs inference on the specified image,
    and extracts the segmented objects (stigmas).
    """
    # Load the model
    model = YOLO(model_checkpoint)

    # Load and convert the image to RGB
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_num = os.path.basename(image_path).split(".")[0]
    print(f"Inference for image {image_num}")

    # Run inference
    results = model.predict(image)
    result = results[0]

    # Extract and save segmented objects
    segmented_objects = extract_segmented_objects(
        image_rgb, result, save_dir=save_dir, image_name=image_num
    )
    return segmented_objects


# -------------------------------
# 3. Main Pipeline
# -------------------------------


def main():
    # Set paths and parameters
    stigmas_model_checkpoint = "/home/etaylor/code_projects/thesis/checkpoints/stigmas_segmentation/yolo/fine_tuned/yolov8m-seg_fine_tuned.pt"

    image_path = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_2/images/day_1_2024_12_05/lab/6/IMG_0626.JPG"

    # Use the image file name (without extension) as the image name for segments.ai
    image_name = os.path.basename(image_path).split(".")[0]

    # Directory to save the segmented objects
    save_dir = "/home/etaylor/code_projects/thesis/src/stigmas_detection/segmentation/yolo/extracted_stigmas"
    save_dir = os.path.join(save_dir, image_name)

    # Run inference and extract segmented objects
    segmented_objects = run_inference(image_path, stigmas_model_checkpoint, save_dir)

    print(f"Saved {len(segmented_objects)} stigmas objects to: {save_dir}")


if __name__ == "__main__":
    main()
