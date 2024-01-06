from segments import SegmentsClient, SegmentsClient, SegmentsDataset
from segments.utils import load_image_from_url, load_label_bitmap_from_url, bitmap2file
from extra_deps.fast_labeling_workflow.fast_labeling_utils import visualize

import cv2
import numpy as np
import os

from dotenv import load_dotenv
load_dotenv()

# ---- THIS SEGMENTSAI HANDLER IS DEPRECATED - USED TO WORK WITH SAM MODEL ----


def get_ground_truth(index, dataset, client):
    sample = client.get_samples(dataset)[index]
    label = client.get_label(sample.uuid, labelset="ground-truth")
    total = len(label.attributes.annotations)
    # Load the labels as numpy arrays
    image = load_image_from_url(sample.attributes.image.url)
    instance_bitmap = load_label_bitmap_from_url(
        label.attributes.segmentation_bitmap.url
    )
    semantic_bitmap = instance_bitmap.copy()
    semantic_bitmap[instance_bitmap > 0] = 1
    return image, instance_bitmap, semantic_bitmap, total, sample.name


def build_ground_truth_dict(dataset, client):
    ground_truth_dict = {}
    num_samples = len(client.get_samples(dataset))

    for i in range(num_samples):
        image, instance_bitmap, semantic_bitmap, total, name = get_ground_truth(
            i, dataset, client
        )
        name = name.split(".")[0]
        ground_truth_dict[name] = {
            "image": image,
            "instance_bitmap": instance_bitmap,
            "semantic_bitmap": semantic_bitmap,
            "total_annotations": total,
        }

    return ground_truth_dict



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
