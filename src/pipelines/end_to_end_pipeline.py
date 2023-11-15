import cv2
import numpy as np
import os

from src.segmentation.utils import get_bboxes_from_segmentation, crop_and_store_bboxes
from src.classification.utils import classify_objects
from src.utils.visualization import plot_cut_images, plot_with_bboxes


def segment_and_classify_image(
    image_path,
    segmentation_model,
    save_dir,
    classification_model=None,
    verbose=False,
    plot=False,
):
    if verbose:
        print(f"Reading image from {image_path}")

    # Read the image
    image = cv2.imread(image_path)

    if verbose:
        print("Generating segmentation bitmap using the model")

    # Generate the segmentation_bitmap using your model
    segmentation_bitmap, _ = segmentation_model(image)

    # Check if any segments were found
    if np.all(segmentation_bitmap == 0):
        print(
            "No segments found by the model for image path = {image_path}. Skipping this image."
        )
        return

    if verbose:
        print("Finding bounding boxes")

    # Find bounding boxes
    bboxes = get_bboxes_from_segmentation(segmentation_bitmap)

    if plot:
        if verbose:
            print("Plotting image with bounding boxes")
        plot_with_bboxes(image, bboxes)

    if verbose:
        print("Cutting and saving bounding boxes")

    # Cut and save bounding boxes
    cut_images = crop_and_store_bboxes(image, bboxes, save_dir)

    # Classify objects if a classification model is provided
    classification_results = None
    if classification_model:
        if verbose:
            print("Classifying objects")
        classification_results = classify_objects(cut_images, classification_model)

    # Plotting section
    if plot:
        if cut_images:  # Check if cut_images is not empty
            if verbose:
                print("Plotting cut images")
            plot_cut_images(cut_images)

        if classification_results:  # Check if classification_results is not empty
            if verbose:
                print("Plotting image with bounding boxes and classification results")
            plot_with_bboxes(image, bboxes, classification_results)
        elif (
            not classification_model
        ):  # If no classification model provided, just plot bboxes
            if verbose:
                print("Plotting image with bounding boxes")
            plot_with_bboxes(image, bboxes)


def batch_process_images(
    image_dir,
    segmentation_model,
    base_save_dir,
    classification_model=None,
    verbose=False,
    plot=False,
):
    # Iterate through all files in the directory
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith(
                (".jpg", ".png", ".jpeg")
            ):  # Add more extensions if needed
                image_path = os.path.join(root, file)

                # Create a save directory based on the image's directory and name
                relative_path = os.path.relpath(root, image_dir)
                save_dir = os.path.join(
                    base_save_dir, relative_path, os.path.splitext(file)[0]
                )

                # Create the save directory if it doesn't exist
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                if verbose:
                    print(f"Processing {image_path} and saving to {save_dir}")

                # Run the pipeline for this image
                segment_and_classify_image(
                    image_path,
                    segmentation_model,
                    save_dir,
                    classification_model,
                    verbose=verbose,
                    plot=plot,
                )
