from src.data_preparation.image_loader import read_images_and_names
from src.data_preparation.sharpness_assessment import calculate_sharpness
from src.data_preparation.patch_cutter import cut_images, save_patches
import config

import numpy as np


def preprocessing_pipeline(images_path: str, saving_dir: str, verbose: bool = False, patch_size: int = 500):
    """
    Process images from a specified directory by cutting each image into patches,
    evaluating the sharpness of each patch, filtering out blurry patches, and saving
    the sharp patches to a specified directory.

    Parameters:
    images_path (str): Path to the directory containing the images to be processed.
    saving_dir (str): Directory where the sharp patches will be saved.
    verbose (bool, optional): If True, print progress messages during processing. Defaults to False.

    The processing pipeline includes the following steps for each image:
    1. Load the image from the specified directory.
    2. Cut the image into patches.
    3. Evaluate the sharpness of each patch.
    4. Filter out blurry patches based on sharpness evaluation.
    5. Save the sharp patches to the specified directory.
    """

    images_and_names = read_images_and_names(dir_path=images_path, verbose=verbose)

    # Iterate through each image in the list
    for image, image_name in images_and_names:
        if verbose:
            print(f"Processing image: {image_name}")

        # Cut the image into patches
        patches, _ = cut_images(image, patch_height=patch_size, patch_width=patch_size)
        if verbose:
            print(f"Extracted {len(patches)} patches from {image_name}")

        # Calculate the sharpness and monochromatic values for each patch
        sharpness_values = [calculate_sharpness(patch) for patch in patches]
        if verbose:
            print(f"Calculated sharpness values for patches of {image_name}")

        # Get the average sharpness and monochromatic values for the patches
        avg_sharpness = np.mean(sharpness_values)

        # Preprocess each patch
        preprocessed_patches = [
            patch
            for patch, sharpness in zip(patches, sharpness_values)
            if sharpness > avg_sharpness
        ]
        if verbose:
            print(
                f"Selected {len(preprocessed_patches)} preprocessed patches for {image_name}"
            )

        # Save the preprocessed patches to the specified directory
        save_patches(image_name, preprocessed_patches, saving_dir)
        if verbose:
            print(f"Saved preprocessed patches of {image_name} to {saving_dir}")


if __name__ == "__main__":

    week = 'week8'
    zoom_type = '3xr'
    patch_size = 512
    source_images_path = config.get_raw_image_path(week, zoom_type)
    saving_images_path = config.get_processed_cannabis_image_path(week, zoom_type)
    verbose = True
    preprocessing_pipeline(
        images_path=source_images_path, saving_dir=saving_images_path, verbose=verbose, patch_size=patch_size
    )
