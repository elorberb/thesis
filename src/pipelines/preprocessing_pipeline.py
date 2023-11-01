from src.data_preparation.image_loader import read_images_and_names
from src.data_preparation.patch_cutter import preprocess_patches
from src.data_preparation import sharpness_assessment
from src.data_preparation.patch_cutter import cut_images, save_patches

import numpy as np


def preprocessing_pipeline(images_path, saving_dir, verbose=False):
    """
    Preprocess images from a given directory and save the processed patches.

    Parameters:
    - images_path (str): Path to the directory containing the images to preprocess.
    - saving_dir (str): Directory where the preprocessed patches will be saved.
    - verbose (bool, optional): If True, print progress messages. Defaults to False.
    """

    trichomes_images = read_images_and_names(dir_path=images_path, verbose=verbose)

    # Iterate through each image in the list
    for (image, image_name) in trichomes_images:
        if verbose:
            print(f"Processing image: {image_name}")

        # Cut the image into patches
        patches, _ = cut_images(image)
        if verbose:
            print(f"Extracted {len(patches)} patches from {image_name}")

        # Calculate the sharpness and monochromatic values for each patch
        sharpness_values = [sharpness_assessment.calculate_sharpness(patch) for patch in patches]
        if verbose:
            print(f"Calculated sharpness values for patches of {image_name}")

        # Get the average sharpness and monochromatic values for the patches
        avg_sharpness = np.mean(sharpness_values)

        # Preprocess each patch
        preprocessed_patches = [patch for patch, sharpness in zip(patches, sharpness_values) if sharpness > avg_sharpness]
        if verbose:
            print(f"Selected {len(preprocessed_patches)} preprocessed patches for {image_name}")

        # Save the preprocessed patches to the specified directory
        save_patches(image_name, preprocessed_patches, saving_dir)
        if verbose:
            print(f"Saved preprocessed patches of {image_name} to {saving_dir}")

    
    


if __name__ == '__main__':
    images_path = "/sise/home/etaylor/images/raw_images/week9_15_06_2023/3x_regular"
    saving_dir = "/sise/home/etaylor/images/processed_images/cannabis_patches/week9_3xzoom_regular_v3"
    verbose = True
    preprocessing_pipeline(images_path=images_path, saving_dir=saving_dir, verbose=verbose)