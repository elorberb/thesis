import os
import cv2
import pandas as pd
# from deprecated import deprecated

# @deprecated(reason="Use pad_and_cut_images function instead")
# def cut_images(image, patch_height=512, patch_width=512):
#     patches_with_coords = []
#     height, width, _ = image.shape

#     for i in range(0, height - patch_height + 1, patch_height):
#         for j in range(0, width - patch_width + 1, patch_width):
#             patch = image[i: i + patch_height, j: j + patch_width]
#             if patch.shape[0] == patch_height and patch.shape[1] == patch_width:
#                 patches_with_coords.append((patch, (i, j)))  # Append (patch, (i, j))

#     return patches_with_coords


def cut_images(image, patch_size=512):
    """ Cut the image into patches of the specified size and return them with their coordinates. """
    # Calculate the required padding
    pad_height = (patch_size - image.shape[0] % patch_size) % patch_size
    pad_width = (patch_size - image.shape[1] % patch_size) % patch_size

    # Pad the image to make it divisible by the patch size
    padded_image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Then cut the padded image into patches
    patches_with_coords = []
    for i in range(0, padded_image.shape[0], patch_size):
        for j in range(0, padded_image.shape[1], patch_size):
            patch = padded_image[i: i + patch_size, j: j + patch_size]
            patches_with_coords.append((patch, (i, j)))

    return patches_with_coords


def save_patches(image_name, patches, saving_dir, csv_file_path):
    """
    Save the patches to the specified directory and update the metadata CSV file using Pandas.

    Parameters:
    image_name (str): Name of the original image.
    patches (list): List of tuples containing patches and their coordinates.
    saving_dir (str): Directory where the patches will be saved.
    csv_file_path (str): Path to the CSV file where metadata will be stored.
    """
    # Make sure the saving directory exists
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
        
    # Initialize an empty DataFrame for new metadata
    new_metadata_df = pd.DataFrame(columns=['patch_name', 'y', 'x'])

    for i, (patch, coords) in enumerate(patches):
        patch_filename = f"{image_name}_p{i}.png"
        patch_path = os.path.join(saving_dir, patch_filename)

        # Save the patch image
        cv2.imwrite(patch_path, patch)
        
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
