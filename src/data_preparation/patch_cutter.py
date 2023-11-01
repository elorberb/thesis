import os
import cv2
import numpy as np

from src.data_preparation import sharpness_assessment

def cut_images(image, patch_height=500, patch_width=500):
    patches = []
    patches_with_coords = []
    height, width, _ = image.shape

    for i in range(0, height - patch_height + 1, patch_height):
        for j in range(0, width - patch_width + 1, patch_width):
            patch = image[i:i + patch_height, j:j + patch_width]
            # Check if the patch size is as expected
            if patch.shape[0] != patch_height or patch.shape[1] != patch_width:
                continue
            patches.append(patch)
            patches_with_coords.append((patch, i, j))
    return patches, patches_with_coords


def save_patches(image_name, patches, dir_path):
    """
    Saves the given patches with the original image name as the prefix in the specified directory.

    Parameters:
        image_name (str): the original image name to use as the prefix for the file names
        patches (list): a list of patches (images_and_names) to save
        dir_path (str): the directory to save the patches in

    Returns:
        None
    """
    # Create the image name subdirectory if it does not exist
    subdir_path = os.path.join(dir_path, image_name)
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)

    for i, patch in enumerate(patches):
        # Use the image name subdirectory as the file path
        file_path = os.path.join(subdir_path, f"{image_name}_p{i}.png")
        cv2.imwrite(file_path, patch)


def apply_function_to_images(images_and_names, func):
    modified_images_and_names = []
    for image, name in images_and_names:
        modified_image = func(image)
        modified_images_and_names.append((modified_image, name))
    return modified_images_and_names

def preprocess_patches(saving_dir_path, trichomes_images, verbose=False):
    """
    Preprocesses the patches of trichome images to remove monochromatic and blurry patches.

    Parameters:
    - saving_dir_path (str): The path to the directory where the preprocessed patches will be saved.
    - trichomes_images (list): A list of tuples, where each tuple consists of a trichome image and the image name.
    - verbose (bool): Flag to print information about the ongoing operations. Defaults to False.

    Returns:
    - None
    """
    # Preprocess the patches of each image
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
        save_patches(image_name, preprocessed_patches, saving_dir_path)
        if verbose:
            print(f"Saved preprocessed patches of {image_name} to {saving_dir_path}")
