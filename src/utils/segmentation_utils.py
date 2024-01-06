import numpy as np
import cv2
import os


def get_bboxes_from_segmentation(segmentation_bitmap, margin=5):
    unique_labels = np.unique(segmentation_bitmap)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background (0)
    bboxes = {}

    for label in unique_labels:
        label_mask = segmentation_bitmap == label
        coords = np.column_stack(np.where(label_mask))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        y_min = max(y_min - margin, 0)
        x_min = max(x_min - margin, 0)
        y_max = min(y_max + margin, segmentation_bitmap.shape[0])
        x_max = min(x_max + margin, segmentation_bitmap.shape[1])

        bboxes[f"bbox_{label}"] = (y_min, y_max, x_min, x_max)

    return bboxes


def crop_and_store_bboxes(image, bboxes, save_dir):
    cut_images = {}

    for label, (y_min, y_max, x_min, x_max) in bboxes.items():
        cut_image = image[y_min:y_max, x_min:x_max]
        cut_images[label] = cut_image
        save_path = os.path.join(save_dir, f"{label}.png")
        cv2.imwrite(save_path, cut_image)

    return cut_images
