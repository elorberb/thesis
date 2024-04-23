import os
import cv2


def extract_file_name(file_path):
    """
    Extracts the file name from a file path.
    """
    # Split the file path into a list of strings
    parts = file_path.split("/")
    # Get the last element in the list (the file name)
    file_name = parts[-1]
    # Split the file name into a list of strings
    parts = file_name.split(".")
    # Get the first element in the list (the file name without the extension)
    file_name = parts[0]

    return file_name


def read_images(input_path_or_list, func=None, verbose=False):
    """
    Read images from either a directory path or a list of image paths and return them as a dictionary.
    The dictionary's keys are the names of the images, and the values are the NumPy arrays representing the images.

    Parameters:
    - input_path_or_list (str or list): A string path to the directory containing images or a list of image paths.
    - func (function, optional): A function to apply to each image after reading.
    - verbose (bool): Flag to print information about the ongoing operations. Defaults to False.

    Returns:
    - dict: A dictionary with image names as keys and image data as values.
    """
    images = {}

    # Determine if input is a directory path or a list of paths
    if isinstance(input_path_or_list, str):
        # Assume it is a directory path
        image_paths = [os.path.join(input_path_or_list, filename) for filename in os.listdir(input_path_or_list)
                    if filename.endswith((".png", ".jpg", ".PNG", ".JPG", ".jpeg"))]
    elif isinstance(input_path_or_list, list):
        # Assume it is a list of file paths
        image_paths = input_path_or_list
    else:
        raise ValueError("Input must be a directory path or a list of image paths.")

    for file_path in image_paths:
        filename = os.path.basename(file_path)
        image = cv2.imread(file_path)

        if func is not None:
            image = func(image)

        images[filename] = image

        if verbose:
            print(f"Processed image: {filename}")

    return images
