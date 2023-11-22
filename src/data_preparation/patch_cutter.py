import os
import cv2
import pandas as pd

# Deprecated for pad_and_cut_images function
def cut_images(image, patch_height=512, patch_width=512):
    patches_with_coords = []
    height, width, _ = image.shape

    for i in range(0, height - patch_height + 1, patch_height):
        for j in range(0, width - patch_width + 1, patch_width):
            patch = image[i: i + patch_height, j: j + patch_width]
            if patch.shape[0] == patch_height and patch.shape[1] == patch_width:
                patches_with_coords.append((patch, (i, j)))  # Append (patch, (i, j))

    return patches_with_coords


def pad_and_cut_images(image, patch_height=512, patch_width=512):
    # Calculate the required padding
    pad_height = (patch_height - image.shape[0] % patch_height) % patch_height
    pad_width = (patch_width - image.shape[1] % patch_width) % patch_width

    # Pad the image
    padded_image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Then cut the padded image into patches
    patches_with_coords = []
    for i in range(0, padded_image.shape[0], patch_height):
        for j in range(0, padded_image.shape[1], patch_width):
            patch = padded_image[i: i + patch_height, j: j + patch_width]
            patches_with_coords.append((patch, (i, j)))

    return patches_with_coords


def save_patches_with_metadata(image_name, patches_with_coords, saving_dir, csv_file_path):
    """
    Save the patches to the specified directory and update the metadata CSV file using Pandas.

    Parameters:
    image_name (str): Name of the original image.
    patches_with_coords (list): List of tuples containing patches and their coordinates.
    saving_dir (str): Directory where the patches will be saved.
    csv_file_path (str): Path to the CSV file where metadata will be stored.
    """
    # Initialize an empty DataFrame for new metadata
    new_metadata_df = pd.DataFrame(columns=['patch_name', 'y', 'x'])

    for i, (patch, coords) in enumerate(patches_with_coords):
        patch_filename = f"{image_name}_p{i}.png"
        # ... [code to save patch image] ...
        
        # Add new row to the DataFrame
        new_metadata_df.loc[i] = [patch_filename, coords[0], coords[1]]

    # If the CSV file exists, load it and concatenate the new data, otherwise use new data
    if os.path.exists(csv_file_path):
        existing_df = pd.read_csv(csv_file_path)
        metadata_df = pd.concat([existing_df, new_metadata_df], ignore_index=True)
    else:
        metadata_df = new_metadata_df

    # Save the updated DataFrame to the CSV file
    metadata_df.to_csv(csv_file_path, index=False)



def apply_function_to_images(images_and_names, func):
    modified_images_and_names = []
    for image, name in images_and_names:
        modified_image = func(image)
        modified_images_and_names.append((modified_image, name))
    return modified_images_and_names
