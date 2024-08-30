from src.data_preparation.image_loader import read_images
from src.data_preparation.sharpness_assessment import calculate_sharpness
from src.data_preparation.patch_cutter import cut_images, save_patches
import config
import logging
import numpy as np
import cv2

# Setup logging configuration
logging.basicConfig(level=logging.INFO)


def filter_sharp_patches(patches):
    # Filter out blurry patches based on sharpness evaluation
    sharp_patches = [
        (patch, coords)
        for patch, coords in patches
        if calculate_sharpness(patch)
        > np.mean([calculate_sharpness(p) for p, _ in patches])
    ]

    return sharp_patches


def preprocess_single_image(image_or_path, image_name, patch_size=512, verbose=False):
    """
    Process a single image by cutting it into patches, evaluating sharpness, and optionally filtering out blurry patches.
    The function can accept either an image array or a path to an image file.

    Parameters:
    image_or_path (ndarray or str): The image array or path to the image to process.
    image_name (str): The name of the image for logging purposes.
    patch_size (int): The size of each patch.
    verbose (bool): If True, print detailed logs.

    Returns:
    list: A list of sharp patches.
    """
    if isinstance(image_or_path, str):
        # Load the image from the path if a string is provided
        image = cv2.imread(image_or_path)
        if image is None:
            logging.error(f"Failed to load image from path: {image_or_path}")
            return []
        if verbose:
            logging.info(f"Loaded image from path: {image_or_path}")
    else:
        # Assume the input is already an image array
        image = image_or_path

    if verbose:
        logging.info(f"Processing image: {image_name}")

    # Cut the image into patches and get their coordinates
    patches = cut_images(image, patch_size=patch_size)
    if verbose:
        print(f"Extracted {len(patches)} patches from {image_name}")

    # Filter out blurry patches based on sharpness evaluation
    sharp_patches = filter_sharp_patches(patches)
    if verbose:
        logging.info(f"Filtered {len(sharp_patches)} sharp patches for {image_name}")

    return sharp_patches


def preprocessing_pipeline(images_source: str, verbose=False, patch_size=512, **kwargs):
    """Process images by cutting them into patches, evaluating sharpness, and filtering out blurry patches."""

    images = read_images(input_path_or_list=images_source, verbose=verbose)
    images_patches = {}
    for image_name, image in images.items():
        sharp_patches = preprocess_single_image(
            image, image_name, patch_size=patch_size, verbose=verbose
        )
        (
            logging.info(
                f"Filtered {len(sharp_patches)} sharp patches for {image_name}"
            )
            if verbose
            else None
        )

        if kwargs.get("saving_dir", None) and kwargs.get("csv_file_path", None):
            # Save the sharp patches to the specified directory and update the CSV file
            save_patches(
                image_name,
                sharp_patches,
                kwargs.get("saving_dir", None),
                kwargs.get("csv_file_path", None),
            )
            if verbose:
                print(
                    f"Saved sharp patches of {image_name} to {kwargs.get('saving_dir', None)}"
                )

        images_patches[image_name] = sharp_patches

    return images_patches


if __name__ == "__main__":

    week = "week9"
    zoom_type = "3xr"
    patch_size = config.CANNABIS_PATCH_SIZE
    source_images_path = config.get_raw_image_path(week, zoom_type)
    saving_images_path = config.get_processed_cannabis_image_path(week, zoom_type)
    verbose = True
    csv_file_path = "metadata/cannabis_patches_metadata.csv"
    preprocessing_pipeline(
        images_source=source_images_path,
        saving_dir=saving_images_path,
        csv_file_path=csv_file_path,
        verbose=verbose,
        patch_size=patch_size,
    )
