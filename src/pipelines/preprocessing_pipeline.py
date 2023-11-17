from src.data_preparation.image_loader import read_images_and_names
from src.data_preparation.sharpness_assessment import calculate_sharpness
from src.data_preparation.patch_cutter import cut_images, save_patches_with_metadata
import config

import numpy as np


def preprocessing_pipeline(images_path: str, saving_dir: str, csv_file_path: str, verbose: bool = False, patch_size: int = 512):
    """
    Process images from a specified directory by cutting each image into patches,
    evaluating the sharpness of each patch, filtering out blurry patches, and saving
    the sharp patches along with their coordinates to a specified directory and updating a CSV file.

    Parameters:
    images_path (str): Path to the directory containing the images to be processed.
    saving_dir (str): Directory where the sharp patches will be saved.
    csv_file_path (str): Path to the CSV file where patch metadata will be stored.
    verbose (bool, optional): If True, print progress messages during processing. Defaults to False.
    patch_size (int): Size of each square patch (default is 512).
    """

    images_and_names = read_images_and_names(dir_path=images_path, verbose=verbose)

    for image, image_name in images_and_names:
        if verbose:
            print(f"Processing image: {image_name}")

        # Cut the image into patches and get their coordinates
        patches_with_coords = cut_images(image, patch_height=patch_size, patch_width=patch_size)
        if verbose:
            print(f"Extracted {len(patches_with_coords)} patches from {image_name}")

        # Filter out blurry patches based on sharpness evaluation
        sharp_patches_with_coords = [
            (patch, coords) for patch, coords in patches_with_coords
            if calculate_sharpness(patch) > np.mean([calculate_sharpness(p) for p, _ in patches_with_coords])
        ]
        if verbose:
            print(f"Selected {len(sharp_patches_with_coords)} sharp patches for {image_name}")

        # Save the sharp patches to the specified directory and update the CSV file
        save_patches_with_metadata(image_name, sharp_patches_with_coords, saving_dir, csv_file_path)
        if verbose:
            print(f"Saved sharp patches of {image_name} to {saving_dir}")


if __name__ == "__main__":

    week = 'week5'
    zoom_type = '3xr'
    patch_size = config.CANNABIS_PATCH_SIZE
    source_images_path = config.get_raw_image_path(week, zoom_type)
    saving_images_path = config.get_processed_cannabis_image_path(week, zoom_type)
    verbose = True
    csv_file_path = 'metadata/cannabis_patches_metadata.csv'
    preprocessing_pipeline(
        images_path=source_images_path, 
        saving_dir=saving_images_path, 
        csv_file_path=csv_file_path,
        verbose=verbose, 
        patch_size=patch_size
    )
