from segments import SegmentsClient, SegmentsClient, SegmentsDataset
from segments.utils import load_image_from_url, load_label_bitmap_from_url, bitmap2file
from extra_deps.fast_labeling_workflow.fast_labeling_utils import visualize

import cv2
import numpy as np
import os

from dotenv import load_dotenv
load_dotenv()

# ---- THIS SEGMENTSAI HANDLER IS DEPRECATED - USED TO WORK WITH SAM MODEL ----


def setup_connection_segmentsai():
    api_key = os.getenv("SEGMENTS_API_KEY")
    client = SegmentsClient(api_key)
    return client


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


def create_dataset(client, name, description, task_type, task_attributes):
    """
    Create a new dataset using the Segments AI API.

    Example:
        Dataset details for a generic dataset:

        name = "trichomes_patches_week8_zoom3_regular_IMG_1802"
        description = "Trichomes patches of image IMG_1802 - purple flower."
        task_type = "segmentation-bitmap"
        task_attributes = {
            "format_version": "0.1",
            "categories": [
                {"name": "trichome", "id": 1},
            ]
        }

        #Create the dataset:

        dataset = create_dataset(name, description, task_type, task_attributes)
        print(dataset)
    """
    dataset = client.add_dataset(name, description, task_type, task_attributes)
    return dataset


def upload_images_to_segments(
    client, dataset_name, image_directory, file_extension=".jpg"
):
    """
    Upload images from a local directory to Segments.ai dataset.
    """
    # List all image files in the directory with the specified file extension
    image_files = [f for f in os.listdir(image_directory) if f.endswith(file_extension)]

    # Upload the images to Segments.ai
    for image_file in image_files:
        # Construct the full path to the image file
        image_path = os.path.join(image_directory, image_file)

        # Upload the image file to Segments.ai's asset storage
        with open(image_path, "rb") as f:
            asset = client.upload_asset(
                f, image_file
            )  # Use the image_file as the filename for the asset

        # Get the URL of the uploaded asset
        image_url = asset.url

        # Create the attributes for the dataset using the image URL
        attributes = {"image": {"url": image_url}}

        # Add the sample to the dataset using the image URL
        sample = client.add_sample(dataset_name, image_file, attributes)
        print(f"Sample {image_file} added: {sample}")


def visualize_dataset(client, dataset_name, release_version="v0.1"):
    """
    Visualize samples in a Segments.ai dataset.
    """
    # Initialize a dataset from the release file
    release = client.get_release(dataset_name, release_version)
    dataset = SegmentsDataset(release)

    # Visualize a few samples in the dataset
    for sample in dataset:
        try:
            # existing code
            visualize(sample["image"], sample["segmentation_bitmap"])
        except TypeError as e:
            print(f"Skipping sample {sample['name']} due to TypeError: {e}")


def generate_and_upload_predictions(
    client, dataset_name_test, model, visualize_flag=False
):
    """
    Generate and upload label predictions for the unlabeled images.
    """

    # Initialize a new dataset, this time containing only unlabeled images
    # Initialize a dataset from the release file
    release = client.get_release(dataset_name_test, "v0.1")
    dataset = SegmentsDataset(release, labelset="ground-truth", filter_by="unlabeled")

    for sample in dataset:
        # Generate label predictions
        image = sample["image"]
        segmentation_bitmap, annotations = model(image)

        if visualize_flag:
            # Visualize the predictions
            visualize(image, segmentation_bitmap)
            print(annotations)

        # Upload the predictions to Segments.ai
        f = bitmap2file(segmentation_bitmap)
        asset = client.upload_asset(f, "label.png")
        attributes = {
            "format_version": "0.1",
            "annotations": annotations,
            "segmentation_bitmap": {"url": asset.url},
        }
        client.add_label(
            sample["uuid"], "ground-truth", attributes, label_status="PRELABELED"
        )


def copy_samples_and_annotations(
    client, src_dataset_identifier, dest_dataset_identifier, verbose=False
):
    """
    Copy all samples and their annotations from one dataset to another.
    """

    # Get samples from the source dataset
    src_samples = client.get_samples(src_dataset_identifier)

    if verbose:
        print(f"Found {len(src_samples)} samples in source dataset.")

    # Loop through each sample in the source dataset
    for i, sample in enumerate(src_samples):
        if verbose:
            print(f"Processing sample {i + 1}/{len(src_samples)}: {sample.name}")

        # Get the label for the sample from the source dataset
        label = client.get_label(sample.uuid, labelset="ground-truth")

        if label:
            if verbose:
                print(f"  - Found label for sample {sample.name}. Copying...")

            # Add the sample to the destination dataset
            new_sample = client.add_sample(
                dest_dataset_identifier, sample.name, sample.attributes
            )

            # Add the label to the new sample in the destination dataset
            client.add_label(new_sample.uuid, "ground-truth", label.attributes)

            if verbose:
                print(f"  - Successfully copied sample {sample.name} and its label.")
        else:
            if verbose:
                print(f"  - No label found for sample {sample.name}. Skipping...")


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
